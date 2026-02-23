use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Write};

#[derive(Debug)]
struct OHLC {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    pattern: HashSet<String>,
}

fn parse_ohlc(line: &str) -> Option<OHLC> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 5 {
        eprintln!("Warning: Skipping line with incorrect number of parts: {}", line);
        return None;
    }

    let date = parts[0].trim().to_string();
    let open: f64 = parts[1].trim().parse().ok()?;
    let high: f64 = parts[2].trim().parse().ok()?;
    let low: f64 = parts[3].trim().parse().ok()?;
    let close: f64 = parts[4].trim().parse().ok()?;

    Some(OHLC {
        date,
        open,
        high,
        low,
        close,
        pattern: HashSet::new(),
    })
}

fn get_price(ohlc: &OHLC) -> f64 {
    ohlc.open.max(ohlc.close)
}

fn get_min_price(ohlc: &OHLC) -> f64 {
    ohlc.open.min(ohlc.close)
}

fn compute_group_pivots(data: &[OHLC], start: usize, end: usize) -> Option<((usize, f64), (usize, f64))> {
    if start >= data.len() || end > data.len() || start >= end {
        return None;
    }
    let mut max_idx = start;
    let mut max_price = get_price(&data[start]);
    let mut min_idx = start;
    let mut min_price = get_min_price(&data[start]);

    for i in start + 1..end {
        let curr_price = get_price(&data[i]);
        let curr_min_price = get_min_price(&data[i]);
        if curr_price > max_price {
            max_price = curr_price;
            max_idx = i;
        }
        if curr_min_price < min_price {
            min_price = curr_min_price;
            min_idx = i;
        }
    }
    Some(((max_idx, max_price), (min_idx, min_price)))
}

fn analyze_patterns(data: &mut Vec<OHLC>, timeframe: &str) {
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    let mut minima: Vec<(usize, f64)> = Vec::new();
    let mut log_entries: Vec<String> = Vec::new();
    let mut pattern_id = 1;

    // Set thresholds based on timeframe
    let (gap_threshold, min_decline, min_recovery) = match timeframe {
        "weekly" => (0.015, 0.0075, 0.015),  // 1.5% gap, 0.75% min decline, 1.5% recovery
        "monthly" => (0.02, 0.01, 0.02),     // 2% gap, 1% min decline, 2% recovery
        _ => (0.005, 0.005, 0.01),         // Default daily values
    };

    if data.len() < 3 {
        return;
    }
    for i in 1..(data.len() - 1) {
        let prev_price = get_price(&data[i - 1]);
        let curr_price = get_price(&data[i]);
        let next_price = get_price(&data[i + 1]);
        let prev_min_price = get_min_price(&data[i - 1]);
        let curr_min_price = get_min_price(&data[i]);
        let next_min_price = get_min_price(&data[i + 1]);

        if curr_price >= prev_price && curr_price >= next_price {
            maxima.push((i, curr_price));
        }
        if curr_min_price <= prev_min_price && curr_min_price <= next_min_price {
            minima.push((i, curr_min_price));
        }
    }

    // Nearest Neighbor: Cup (Slope-Based)
    for i in 0..maxima.len() {
        let (x1_idx, x1) = maxima[i];
        if let Some(&(m1_idx, m1)) = minima.iter().find(|&&(idx, _)| idx > x1_idx) {
            if m1 < x1 * (1.0 - min_decline) && m1_idx - x1_idx >= 5 {
                let decline_slope = (m1 - x1) / (m1_idx - x1_idx) as f64;
                if decline_slope >= 0.0 {
                    continue;
                }
                // Search for the highest bar touching X1 within a window
                let window = (m1_idx + 5).max(m1_idx + 3 * (m1_idx - x1_idx));
                let mut x2_idx = None;
                let mut x2_price = 0.0;
                for j in m1_idx + 5..data.len().min(window) {
                    let curr_price = data[j].close.max(data[j].high);
                    if curr_price >= x1 * (1.0 - min_recovery) && curr_price > x2_price {
                        x2_idx = Some(j);
                        x2_price = curr_price;
                    }
                }
                if let Some(x2_idx) = x2_idx {
                    let x2 = x2_price;
                    let recovery_slope = (x2 - m1) / (x2_idx - m1_idx) as f64;
                    if recovery_slope <= 0.0 {
                        continue;
                    }
                    let indices = vec![x1_idx, m1_idx, x2_idx];
                    let labels = vec![
                        format!("Cup_X1_P{}_NN", pattern_id),
                        format!("Cup_M1_P{}_NN", pattern_id),
                        format!("Cup_X2_P{}_NN", pattern_id),
                    ];
                    for (idx, label) in indices.iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.clone());
                    }
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "Cup_X1", x1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "Cup_M1", m1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "Cup_X2", x2));
                    pattern_id += 1;
                }
            }
        }
    }

    // Nearest Neighbor: Cup and Handle
    for i in 0..maxima.len() {
        let (x1_idx, x1) = maxima[i];
        if let Some(&(m1_idx, m1)) = minima.iter().find(|&&(idx, _)| idx > x1_idx) {
            if m1 < x1 * (1.0 - gap_threshold) {
                // Find the highest max after M1 within a window
                let window = (m1_idx + 5).max(m1_idx + 3 * (m1_idx - x1_idx));
                let mut x2_idx = None;
                let mut x2_price = 0.0;
                for j in m1_idx + 5..data.len().min(window) {
                    let curr_price = get_price(&data[j]);
                    if curr_price >= x1 * (1.0 - gap_threshold) && curr_price <= x1 * (1.0 + gap_threshold) && curr_price > x2_price {
                        if j > 1 && j < data.len() - 1 {
                            let prev_price = get_price(&data[j - 1]);
                            let next_price = get_price(&data[j + 1]);
                            if curr_price >= prev_price && curr_price >= next_price {
                                x2_idx = Some(j);
                                x2_price = curr_price;
                            }
                        }
                    }
                }
                if let Some(x2_idx) = x2_idx {
                    let x2 = x2_price;
                    if let Some(&(m2_idx, m2)) = minima.iter().find(|&&(idx, _)| idx > x2_idx) {
                        if m2 < x2 * 0.999 && m2 > m1 {
                            if let Some(&(x3_idx, x3)) = maxima.iter().find(|&&(idx, _)| idx > m2_idx) {
                                if x3 >= x2 * 0.995 && x3 <= x2 * 1.005 {
                                    let indices = vec![x1_idx, m1_idx, x2_idx, m2_idx, x3_idx];
                                    let labels = vec![
                                        format!("CupAndHandle_X1_P{}_NN", pattern_id),
                                        format!("CupAndHandle_M1_P{}_NN", pattern_id),
                                        format!("CupAndHandle_X2_P{}_NN", pattern_id),
                                        format!("CupAndHandle_M2_P{}_NN", pattern_id),
                                        format!("CupAndHandle_X3_P{}_NN", pattern_id),
                                    ];
                                    for (idx, label) in indices.iter().zip(labels.iter()) {
                                        data[*idx].pattern.insert(label.clone());
                                    }
                                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "CupAndHandle_X1", x1));
                                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "CupAndHandle_M1", m1));
                                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "CupAndHandle_X2", x2));
                                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "CupAndHandle_M2", m2));
                                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x3_idx].date, "CupAndHandle_X3", x3));
                                    pattern_id += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Nearest Neighbor: Rising Wedge
    for i in 0..maxima.len() {
        let (m1_idx, m1) = maxima[i];
        if let Some(&(x1_idx, x1)) = minima.iter().find(|&&(idx, _)| idx > m1_idx) {
            if let Some(&(m2_idx, m2)) = maxima.iter().find(|&&(idx, _)| idx > x1_idx) {
                if let Some(&(x2_idx, x2)) = minima.iter().find(|&&(idx, _)| idx > m2_idx) {
                    if let Some(&(m3_idx, m3)) = maxima.iter().find(|&&(idx, _)| idx > x2_idx) {
                        if x1 > m1 * 1.001 && m2 > m1 * 1.001 && x2 > x1 * 1.001 && m3 > m2 * 1.001
                            && (x2 - m3) < (x1 - m1)
                        {
                            let indices = vec![m1_idx, x1_idx, m2_idx, x2_idx, m3_idx];
                            let labels = vec![
                                format!("RisingWedge_M1_P{}_NN", pattern_id),
                                format!("RisingWedge_X1_P{}_NN", pattern_id),
                                format!("RisingWedge_M2_P{}_NN", pattern_id),
                                format!("RisingWedge_X2_P{}_NN", pattern_id),
                                format!("RisingWedge_M3_P{}_NN", pattern_id),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone());
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "RisingWedge_M1", m1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "RisingWedge_X1", x1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "RisingWedge_M2", m2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "RisingWedge_X2", x2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m3_idx].date, "RisingWedge_M3", m3));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Nearest Neighbor: Falling Wedge
    for i in 0..maxima.len() {
        let (x1_idx, x1) = maxima[i];
        if let Some(&(m1_idx, m1)) = minima.iter().find(|&&(idx, _)| idx > x1_idx) {
            if let Some(&(x2_idx, x2)) = maxima.iter().find(|&&(idx, _)| idx > m1_idx) {
                if let Some(&(m2_idx, m2)) = minima.iter().find(|&&(idx, _)| idx > x2_idx) {
                    if let Some(&(x3_idx, x3)) = maxima.iter().find(|&&(idx, _)| idx > m2_idx) {
                        if m1 < x1 * 0.999 && x2 < x1 * 0.999 && m2 < m1 * 0.999 && x3 < x2 * 0.999
                            && (x3 - m2) < (x1 - m1)
                        {
                            let indices = vec![x1_idx, m1_idx, x2_idx, m2_idx, x3_idx];
                            let labels = vec![
                                format!("FallingWedge_X1_P{}_NN", pattern_id),
                                format!("FallingWedge_M1_P{}_NN", pattern_id),
                                format!("FallingWedge_X2_P{}_NN", pattern_id),
                                format!("FallingWedge_M2_P{}_NN", pattern_id),
                                format!("FallingWedge_X3_P{}_NN", pattern_id),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone());
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "FallingWedge_X1", x1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "FallingWedge_M1", m1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "FallingWedge_X2", x2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "FallingWedge_M2", m2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x3_idx].date, "FallingWedge_X3", x3));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Nearest Neighbor: Dead Cat Bounce
    for i in 0..maxima.len() {
        let (x1_idx, x1) = maxima[i];
        if let Some(&(m1_idx, m1)) = minima.iter().find(|&&(idx, _)| idx > x1_idx) {
            if m1 < x1 * (1.0 - gap_threshold) {
                for j in m1_idx + 1..data.len().min(m1_idx + 4) {
                    if data[j].close > data[j].open && get_price(&data[j]) < x1 {
                        let x2_idx = j;
                        let x2 = get_price(&data[j]);
                        let green_count = (m1_idx + 1..=j).filter(|&k| data[k].close > data[k].open).count();
                        if green_count > 3 {
                            break;
                        }
                        if let Some(&(m2_idx, m2)) = minima.iter().find(|&&(idx, _)| idx > x2_idx) {
                            if m2 < x2 * (1.0 - gap_threshold) {
                                let indices = vec![x1_idx, m1_idx, x2_idx, m2_idx];
                                let labels = vec![
                                    format!("DeadCatBounce_X1_P{}_NN", pattern_id),
                                    format!("DeadCatBounce_M1_P{}_NN", pattern_id),
                                    format!("DeadCatBounce_X2_P{}_NN", pattern_id),
                                    format!("DeadCatBounce_M2_P{}_NN", pattern_id),
                                ];
                                for (idx, label) in indices.iter().zip(labels.iter()) {
                                    data[*idx].pattern.insert(label.clone());
                                }
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "DeadCatBounce_X1", x1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "DeadCatBounce_M1", m1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "DeadCatBounce_X2", x2));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "DeadCatBounce_M2", m2));
                                pattern_id += 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Grouping Approach
    let period_groups = vec![10, 20, 30, 60, 90, 183, 366];
    for &period in &period_groups {
        let mut group_maxima: Vec<(usize, f64)> = Vec::new();
        let mut group_minima: Vec<(usize, f64)> = Vec::new();

        let mut start = 0;
        while start < data.len() {
            let end = (start + period).min(data.len());
            if let Some(((max_idx, max_price), (min_idx, min_price))) =
                compute_group_pivots(data, start, end)
            {
                group_maxima.push((max_idx, max_price));
                group_minima.push((min_idx, min_price));
            }
            start += period;
        }

        // Grouping: Cup (Slope-Based)
        for i in 0..group_maxima.len().saturating_sub(1) {
            let x1 = group_maxima[i];
            let m1 = group_minima[i + 1];
            if m1.1 < x1.1 * 0.995 && m1.0 - x1.0 >= 5 {
                let decline_slope = (m1.1 - x1.1) / (m1.0 - x1.0) as f64;
                if decline_slope >= 0.0 {
                    continue;
                }
                let mut x2 = None;
                let mut x2_price = 0.0;
                for j in i + 2..group_maxima.len() {
                    let curr_max = group_maxima[j];
                    if curr_max.1 >= x1.1 * 0.99 && curr_max.1 > x2_price && curr_max.0 - x1.0 <= period * 3 {
                        x2 = Some(curr_max);
                        x2_price = curr_max.1;
                    }
                }
                if let Some(x2) = x2 {
                    let recovery_slope = (x2.1 - m1.1) / (x2.0 - m1.0) as f64;
                    if recovery_slope <= 0.0 || x2.0 - m1.0 < 5 {
                        continue;
                    }
                    let indices = vec![x1.0, m1.0, x2.0];
                    let labels = vec![
                        format!("Cup_X1_P{}_{}", pattern_id, period),
                        format!("Cup_M1_P{}_{}", pattern_id, period),
                        format!("Cup_X2_P{}_{}", pattern_id, period),
                    ];
                    for (idx, label) in indices.iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.clone());
                    }
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "Cup_X1", x1.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "Cup_M1", m1.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "Cup_X2", x2.1));
                    pattern_id += 1;
                }
            }
        }

        // Grouping: Cup and Handle
        for i in 0..group_maxima.len().saturating_sub(4) {
            let x1 = group_maxima[i];
            let m1 = group_minima[i + 1];
            if m1.1 < x1.1 * 0.995 {
                let mut x2 = None;
                let mut x2_price = 0.0;
                for j in i + 2..group_maxima.len().min(i + 3 + period / 10) {
                    let curr_max = group_maxima[j];
                    if curr_max.1 >= x1.1 * 0.995 && curr_max.1 <= x1.1 * 1.005 && curr_max.1 > x2_price {
                        x2 = Some(curr_max);
                        x2_price = curr_max.1;
                    }
                }
                if let Some(x2) = x2 {
                    if x2.0 - x1.0 > period * 3 {
                        continue;
                    }
                    let m2 = group_minima.get(i + 3);
                    let x3 = group_maxima.get(i + 4);
                    if let (Some(m2), Some(x3)) = (m2, x3) {
                        if m2.1 < x2.1 * 0.999 && m2.1 > m1.1 && x3.1 >= x2.1 * 0.995 && x3.1 <= x2.1 * 1.005
                            && x3.0 - x1.0 <= period * 5
                        {
                            let indices = vec![x1.0, m1.0, x2.0, m2.0, x3.0];
                            let labels = vec![
                                format!("CupAndHandle_X1_P{}_{}", pattern_id, period),
                                format!("CupAndHandle_M1_P{}_{}", pattern_id, period),
                                format!("CupAndHandle_X2_P{}_{}", pattern_id, period),
                                format!("CupAndHandle_M2_P{}_{}", pattern_id, period),
                                format!("CupAndHandle_X3_P{}_{}", pattern_id, period),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone());
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "CupAndHandle_X1", x1.1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "CupAndHandle_M1", m1.1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "CupAndHandle_X2", x2.1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "CupAndHandle_M2", m2.1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "CupAndHandle_X3", x3.1));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }

        // Grouping: Rising Wedge
        for i in 0..group_maxima.len().saturating_sub(4) {
            let m1 = group_maxima[i];
            let x1 = group_minima[i + 1];
            let m2 = group_maxima[i + 2];
            let x2 = group_minima[i + 3];
            let m3 = group_maxima[i + 4];

            if m3.0 - m1.0 > period * 5 {
                continue;
            }

            if x1.1 > m1.1 * 1.001 && m2.1 > m1.1 * 1.001 && x2.1 > x1.1 * 1.001 && m3.1 > m2.1 * 1.001
                && (x2.1 - m3.1) < (x1.1 - m1.1)
            {
                let indices = vec![m1.0, x1.0, m2.0, x2.0, m3.0];
                let labels = vec![
                    format!("RisingWedge_M1_P{}_{}", pattern_id, period),
                    format!("RisingWedge_X1_P{}_{}", pattern_id, period),
                    format!("RisingWedge_M2_P{}_{}", pattern_id, period),
                    format!("RisingWedge_X2_P{}_{}", pattern_id, period),
                    format!("RisingWedge_M3_P{}_{}", pattern_id, period),
                ];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "RisingWedge_M1", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "RisingWedge_X1", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "RisingWedge_M2", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "RisingWedge_X2", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "RisingWedge_M3", m3.1));
                pattern_id += 1;
            }
        }

        // Grouping: Falling Wedge
        for i in 0..group_maxima.len().saturating_sub(4) {
            let x1 = group_maxima[i];
            let m1 = group_minima[i + 1];
            let x2 = group_maxima[i + 2];
            let m2 = group_minima[i + 3];
            let x3 = group_maxima[i + 4];

            if x3.0 - x1.0 > period * 5 {
                continue;
            }

            if m1.1 < x1.1 * 0.999 && x2.1 < x1.1 * 0.999 && m2.1 < m1.1 * 0.999 && x3.1 < x2.1 * 0.999
                && (x3.1 - m2.1) < (x1.1 - m1.1)
            {
                let indices = vec![x1.0, m1.0, x2.0, m2.0, x3.0];
                let labels = vec![
                    format!("FallingWedge_X1_P{}_{}", pattern_id, period),
                    format!("FallingWedge_M1_P{}_{}", pattern_id, period),
                    format!("FallingWedge_X2_P{}_{}", pattern_id, period),
                    format!("FallingWedge_M2_P{}_{}", pattern_id, period),
                    format!("FallingWedge_X3_P{}_{}", pattern_id, period),
                ];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "FallingWedge_X1", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "FallingWedge_M1", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "FallingWedge_X2", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "FallingWedge_M2", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "FallingWedge_X3", x3.1));
                pattern_id += 1;
            }
        }

        // Grouping: Dead Cat Bounce
        for i in 0..group_maxima.len().saturating_sub(3) {
            let x1 = group_maxima[i];
            let m1 = group_minima[i + 1];
            let x2 = group_maxima[i + 2];
            let m2 = group_minima[i + 3];

            if m2.0 - x1.0 > period * 4 {
                continue;
            }

            let green_count = (m1.0 + 1..=x2.0)
                .filter(|&k| data[k].close > data[k].open)
                .count();
            if m1.1 < x1.1 * 0.99 && x2.1 < x1.1 && green_count <= 3 && green_count > 0
                && m2.1 < x2.1 * 0.99
            {
                let indices = vec![x1.0, m1.0, x2.0, m2.0];
                let labels = vec![
                    format!("DeadCatBounce_X1_P{}_{}", pattern_id, period),
                    format!("DeadCatBounce_M1_P{}_{}", pattern_id, period),
                    format!("DeadCatBounce_X2_P{}_{}", pattern_id, period),
                    format!("DeadCatBounce_M2_P{}_{}", pattern_id, period),
                ];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "DeadCatBounce_X1", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "DeadCatBounce_M1", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "DeadCatBounce_X2", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "DeadCatBounce_M2", m2.1));
                pattern_id += 1;
            }
        }
    }

    let mut log_file = match File::create("../new_patterns_log.csv") {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating log file: {}", e);
            std::process::exit(1);
        }
    };
    writeln!(log_file, "pattern_id,date,component,price").expect("Failed to write log header");
    for entry in log_entries {
        writeln!(log_file, "{}", entry).expect("Failed to write log line");
    }
}

fn process_dataset(input_filename: &str, output_filename: &str, timeframe: &str) {
    let mut data: Vec<OHLC> = Vec::new();
    let file = match File::open(input_filename) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening input file '{}': {}", input_filename, e);
            std::process::exit(1);
        }
    };

    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    if let Err(e) = reader.read_to_string(&mut contents) {
        eprintln!("Error reading input file '{}': {}", input_filename, e);
        std::process::exit(1);
    }

    let mut line_num = 0;
    for line in contents.lines() {
        line_num += 1;
        if line_num == 1 {
            continue;
        }
        let trimmed_line = line.trim();
        if trimmed_line.is_empty() {
            continue;
        }
        if let Some(ohlc) = parse_ohlc(trimmed_line) {
            data.push(ohlc);
        } else {
            eprintln!("Warning: Failed to parse line {}: {}", line_num, line);
        }
    }

    data.sort_by(|a, b| a.date.cmp(&b.date));
    println!("Starting pattern analysis on {} data points for {}...", data.len(), timeframe);
    analyze_patterns(&mut data, timeframe);
    println!("Pattern analysis complete for {}.", timeframe);

    let mut file = match File::create(output_filename) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating output file '{}': {}", output_filename, e);
            std::process::exit(1);
        }
    };

    if let Err(e) = writeln!(file, "date,open,high,low,close,pattern") {
        eprintln!("Error writing header to output file: {}", e);
        std::process::exit(1);
    }

    let mut lines_written = 0;
    for entry in data {
        let pattern_str = if entry.pattern.is_empty() {
            "".to_string()
        } else {
            let mut patterns: Vec<String> = entry.pattern.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };
        if let Err(e) = writeln!(
            file,
            "{},{},{},{},{},{}",
            entry.date,
            entry.open,
            entry.high,
            entry.low,
            entry.close,
            pattern_str
        ) {
            eprintln!("Error writing line to output file: {}", e);
        } else {
            lines_written += 1;
        }
    }

    println!(
        "Processing complete. {} lines written to {}",
        lines_written, output_filename
    );
}

fn main() {
    // Process GBPAUD datasets
    process_dataset("../GBPAUD_daily.csv", "../new_patterns_GBPAUD_daily.csv", "daily");
    process_dataset("../GBPAUD_weekly.csv", "../new_patterns_GBPAUD_weekly.csv", "weekly");
    process_dataset("../GBPAUD_monthly.csv", "../new_patterns_GBPAUD_monthly.csv", "monthly");
    
    // Process NZDJPY datasets
    process_dataset("../NZDJPY_daily.csv", "../new_patterns_NZDJPY_daily.csv", "daily");
    process_dataset("../NZDJPY_weekly.csv", "../new_patterns_NZDJPY_weekly.csv", "weekly");
    process_dataset("../NZDJPY_monthly.csv", "../new_patterns_NZDJPY_monthly.csv", "monthly");
}