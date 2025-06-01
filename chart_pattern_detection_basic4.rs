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
    if parts.len() != 5 {
        return None;
    }

    Some(OHLC {
        date: parts[0].to_string(),
        open: parts[1].parse().ok()?,
        high: parts[2].parse().ok()?,
        low: parts[3].parse().ok()?,
        close: parts[4].parse().ok()?,
        pattern: HashSet::new(),
    })
}

fn is_within_range(a: f64, b: f64, percentage: f64) -> bool {
    let diff = (a - b).abs();
    diff / a.abs() <= percentage
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

fn slopes_are_parallel(
    idx1: usize,
    price1: f64,
    idx2: usize,
    price2: f64,
    idx3: usize,
    price3: f64,
    idx4: usize,
    price4: f64,
    tolerance: f64,
) -> bool {
    let slope1 = (price2 - price1) / (idx2 as f64 - idx1 as f64);
    let slope2 = (price4 - price3) / (idx4 as f64 - idx3 as f64);
    (slope1 - slope2).abs() <= tolerance
}

fn analyze_patterns(data: &mut Vec<OHLC>) {
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    let mut minima: Vec<(usize, f64)> = Vec::new();
    let mut log_entries = Vec::new();
    let mut pattern_id = 1;

    // Nearest Neighbor Pivot Detection (5-bar window, with significance and distance)
    let min_pivot_distance = 3; // Minimum bars between pivots
    let max_pivot_distance = 60; // Maximum bars between bearish flag points (approx. 2-3 months)
    for i in 0..data.len() {
        let curr_price = get_price(&data[i]);
        let curr_min_price = get_min_price(&data[i]);

        let is_max = (i <= 2 || (
            curr_price >= get_price(&data[i - 1]) * 1.001 && 
            curr_price >= get_price(&data[i - 2]) * 1.001
        )) && (i >= data.len() - 2 || (
            curr_price >= get_price(&data[i + 1]) * 1.001 && 
            curr_price >= get_price(&data[i + 2]) * 1.001
        ));

        let is_min = (i <= 2 || (
            curr_min_price <= get_min_price(&data[i - 1]) * 0.999 && 
            curr_min_price <= get_min_price(&data[i - 2]) * 0.999
        )) && (i >= data.len() - 2 || (
            curr_min_price <= get_min_price(&data[i + 1]) * 0.999 && 
            curr_min_price <= get_min_price(&data[i + 2]) * 0.999
        ));

        if is_max && maxima.last().map_or(true, |(last_idx, _)| i - last_idx >= min_pivot_distance) {
            maxima.push((i, curr_price));
        }
        if is_min && minima.last().map_or(true, |(last_idx, _)| i - last_idx >= min_pivot_distance) {
            minima.push((i, curr_min_price));
        }
    }

    // Staircase: X0 → M1 (< X0 by ≥0.1%) → X1 (> M1 by ≥0.1%) → M2 (within 0%-0.3% of X0)
    for i in 0..maxima.len() {
        let x0 = maxima[i];
        if let Some(&m1) = minima.iter().find(|&&(idx, _)| idx > x0.0) {
            if let Some(&x1) = maxima.iter().find(|&&(idx, _)| idx > m1.0) {
                if let Some(&m2) = minima.iter().find(|&&(idx, _)| idx > x1.0) {
                    if (x0.1 - m1.1) / x0.1 >= 0.001
                        && (x1.1 - m1.1) / m1.1 >= 0.001
                        && is_within_range(x0.1, m2.1, 0.003)
                    {
                        let labels = vec![
                            format!("Staircase_X0_P{}_NN ", pattern_id),
                            format!("Staircase_M1_P{}_NN ", pattern_id),
                            format!("Staircase_X1_P{}_NN ", pattern_id),
                            format!("Staircase_M2_P{}_NN ", pattern_id),
                        ];
                        let indices = vec![x0.0, m1.0, x1.0, m2.0];
                        for (idx, label) in indices.iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.clone());
                        }
                        log_entries.push(format!("{},{},{},{}", pattern_id, data[x0.0].date, "Staircase_X0 ", x0.1));
                        log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "Staircase_M1 ", m1.1));
                        log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "Staircase_X1 ", x1.1));
                        log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "Staircase_M2 ", m2.1));
                        pattern_id += 1;
                    }
                }
            }
        }
    }

    // Bullish Flag: X0 → X1 (> X0 by ≥0.5%, flagpole) → M1 → X2 (≤ X1 * 1.005) → M2 (≥ M1 * 0.995) → X3 (≤ X1 * 1.005)
    for i in 0..maxima.len() {
        let x0 = maxima[i];
        if let Some(&x1) = maxima.iter().find(|&&(idx, _)| idx > x0.0) {
            if let Some(&m1) = minima.iter().find(|&&(idx, _)| idx > x1.0) {
                if let Some(&x2) = maxima.iter().find(|&&(idx, _)| idx > m1.0) {
                    if let Some(&m2) = minima.iter().find(|&&(idx, _)| idx > x2.0) {
                        if let Some(&x3) = maxima.iter().find(|&&(idx, _)| idx > m2.0) {
                            let slope_m1_m2 = (m2.1 - m1.1) / (m2.0 as f64 - m1.0 as f64);
                            let slope_x2_x3 = (x3.1 - x2.1) / (x3.0 as f64 - x2.0 as f64);
                            let flagpole_pct = (x1.1 - x0.1) / x0.1;
                            if flagpole_pct >= 0.005
                                && x2.1 <= x1.1 * 1.005
                                && m2.1 >= m1.1 * 0.995
                                && x3.1 <= x1.1 * 1.005
                                && slope_m1_m2 >= -0.016 && slope_m1_m2 <= 0.016
                                && slope_x2_x3 >= -0.016 && slope_x2_x3 <= 0.016
                                && slopes_are_parallel(m1.0, m1.1, m2.0, m2.1, x2.0, x2.1, x3.0, x3.1, 0.008)
                            {
                                let labels = vec![
                                    format!("BullishFlag_X0_P{}_NN ", pattern_id),
                                    format!("BullishFlag_X1_P{}_NN ", pattern_id),
                                    format!("BullishFlag_M1_P{}_NN ", pattern_id),
                                    format!("BullishFlag_X2_P{}_NN ", pattern_id),
                                    format!("BullishFlag_M2_P{}_NN ", pattern_id),
                                    format!("BullishFlag_X3_P{}_NN ", pattern_id),
                                ];
                                let indices = vec![x0.0, x1.0, m1.0, x2.0, m2.0, x3.0];
                                for (idx, label) in indices.iter().zip(labels.iter()) {
                                    data[*idx].pattern.insert(label.clone());
                                }
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x0.0].date, "BullishFlag_X0 ", x0.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "BullishFlag_X1 ", x1.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "BullishFlag_M1 ", m1.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "BullishFlag_X2 ", x2.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "BullishFlag_M2 ", m2.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "BullishFlag_X3 ", x3.1));
                                pattern_id += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Bearish Flag: M0 → M1 (< M0 by ≥0.5%, flagpole) → X1 → M2 (≤ M1 * 1.005) → X2 (≤ X1 * 1.005) → M3 (≤ M1 * 1.005)
    for i in 0..minima.len() {
        let m0 = minima[i];
        if let Some(&m1) = minima.iter().find(|&&(idx, _)| idx > m0.0 && idx <= m0.0 + max_pivot_distance) {
            if let Some(&x1) = maxima.iter().find(|&&(idx, _)| idx > m1.0 && idx <= m0.0 + max_pivot_distance) {
                if let Some(&m2) = minima.iter().find(|&&(idx, _)| idx > x1.0 && idx <= m0.0 + max_pivot_distance) {
                    if let Some(&x2) = maxima.iter().find(|&&(idx, _)| idx > m2.0 && idx <= m0.0 + max_pivot_distance) {
                        if let Some(&m3) = minima.iter().find(|&&(idx, _)| idx > x2.0 && idx <= m0.0 + max_pivot_distance) {
                            let slope_x1_x2 = (x2.1 - x1.1) / (x2.0 as f64 - x1.0 as f64);
                            let slope_m2_m3 = (m3.1 - m2.1) / (m3.0 as f64 - m2.0 as f64);
                            let flagpole_pct = (m0.1 - m1.1) / m0.1;
                            if flagpole_pct >= 0.005
                                && m2.1 <= m1.1 * 1.005
                                && x2.1 <= x1.1 * 1.005
                                && m3.1 <= m1.1 * 1.005
                                && slope_x1_x2 >= -0.016 && slope_x1_x2 <= 0.016
                                && slope_m2_m3 >= -0.016 && slope_m2_m3 <= 0.016
                                && slopes_are_parallel(x1.0, x1.1, x2.0, x2.1, m2.0, m2.1, m3.0, m3.1, 0.008)
                            {
                                let labels = vec![
                                    format!("BearishFlag_M0_P{}_NN ", pattern_id),
                                    format!("BearishFlag_M1_P{}_NN ", pattern_id),
                                    format!("BearishFlag_X1_P{}_NN ", pattern_id),
                                    format!("BearishFlag_M2_P{}_NN ", pattern_id),
                                    format!("BearishFlag_X2_P{}_NN ", pattern_id),
                                    format!("BearishFlag_M3_P{}_NN ", pattern_id),
                                ];
                                let indices = vec![m0.0, m1.0, x1.0, m2.0, x2.0, m3.0];
                                for (idx, label) in indices.iter().zip(labels.iter()) {
                                    data[*idx].pattern.insert(label.clone());
                                }
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m0.0].date, "BearishFlag_M0 ", m0.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "BearishFlag_M1 ", m1.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "BearishFlag_X1 ", x1.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "BearishFlag_M2 ", m2.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "BearishFlag_X2 ", x2.1));
                                log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "BearishFlag_M3 ", m3.1));
                                pattern_id += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Rising Tops: X1 → X2 (> X1 by 0.1%) → X3 (> X2 by 0.1%)
    for i in 0..maxima.len().saturating_sub(2) {
        let x1 = maxima[i];
        let x2 = maxima[i + 1];
        let x3 = maxima[i + 2];
        if (x2.1 - x1.1) / x1.1 >= 0.001 && (x3.1 - x2.1) / x2.1 >= 0.001 {
            let labels = vec![
                format!("RisingTops_X1_P{}_NN ", pattern_id),
                format!("RisingTops_X2_P{}_NN ", pattern_id),
                format!("RisingTops_X3_P{}_NN ", pattern_id),
            ];
            let indices = vec![x1.0, x2.0, x3.0];
            for (idx, label) in indices.iter().zip(labels.iter()) {
                data[*idx].pattern.insert(label.clone());
            }
            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "RisingTops_X1 ", x1.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "RisingTops_X2 ", x2.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "RisingTops_X3 ", x3.1));
            pattern_id += 1;
        }
    }

    // Rising Bottoms: M1 → M2 (> M1 by 0.1%) → M3 (> M2 by 0.1%)
    for i in 0..minima.len().saturating_sub(2) {
        let m1 = minima[i];
        let m2 = minima[i + 1];
        let m3 = minima[i + 2];
        if (m2.1 - m1.1) / m1.1 >= 0.001 && (m3.1 - m2.1) / m2.1 >= 0.001 {
            let labels = vec![
                format!("RisingBottoms_M1_P{}_NN ", pattern_id),
                format!("RisingBottoms_M2_P{}_NN ", pattern_id),
                format!("RisingBottoms_M3_P{}_NN ", pattern_id),
            ];
            let indices = vec![m1.0, m2.0, m3.0];
            for (idx, label) in indices.iter().zip(labels.iter()) {
                data[*idx].pattern.insert(label.clone());
            }
            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "RisingBottoms_M1 ", m1.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "RisingBottoms_M2 ", m2.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "RisingBottoms_M3 ", m3.1));
            pattern_id += 1;
        }
    }

    // Falling Tops: X1 → X2 (< X1 by 0.1%) → X3 (< X2 by 0.1%)
    for i in 0..maxima.len().saturating_sub(2) {
        let x1 = maxima[i];
        let x2 = maxima[i + 1];
        let x3 = maxima[i + 2];
        if (x1.1 - x2.1) / x1.1 >= 0.001 && (x2.1 - x3.1) / x2.1 >= 0.001 {
            let labels = vec![
                format!("FallingTops_X1_P{}_NN ", pattern_id),
                format!("FallingTops_X2_P{}_NN ", pattern_id),
                format!("FallingTops_X3_P{}_NN ", pattern_id),
            ];
            let indices = vec![x1.0, x2.0, x3.0];
            for (idx, label) in indices.iter().zip(labels.iter()) {
                data[*idx].pattern.insert(label.clone());
            }
            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "FallingTops_X1 ", x1.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "FallingTops_X2 ", x2.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "FallingTops_X3 ", x3.1));
            pattern_id += 1;
        }
    }

    // Falling Bottoms: M1 → M2 (< M1 by 0.1%) → M3 (< M2 by 0.1%)
    for i in 0..minima.len().saturating_sub(2) {
        let m1 = minima[i];
        let m2 = minima[i + 1];
        let m3 = minima[i + 2];
        if (m1.1 - m2.1) / m1.1 >= 0.001 && (m2.1 - m3.1) / m2.1 >= 0.001 {
            let labels = vec![
                format!("FallingBottoms_M1_P{}_NN ", pattern_id),
                format!("FallingBottoms_M2_P{}_NN ", pattern_id),
                format!("FallingBottoms_M3_P{}_NN ", pattern_id),
            ];
            let indices = vec![m1.0, m2.0, m3.0];
            for (idx, label) in indices.iter().zip(labels.iter()) {
                data[*idx].pattern.insert(label.clone());
            }
            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "FallingBottoms_M1 ", m1.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "FallingBottoms_M2 ", m2.1));
            log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "FallingBottoms_M3 ", m3.1));
            pattern_id += 1;
        }
    }

    // Grouping Approach
    let period_groups = vec![10, 20, 30, 60, 90, 183, 366];
    for &period in &period_groups {
        let mut group_maxima: Vec<(usize, f64)> = Vec::new();
        let mut group_minima: Vec<(usize, f64)> = Vec::new();

        let mut start = 0;
        while start + period <= data.len() {
            if let Some(((max_idx, max_price), (min_idx, min_price))) = compute_group_pivots(data, start, start + period) {
                let is_max_significant = (0..data.len()).all(|i| {
                    if i == max_idx || i < start || i >= start + period {
                        true
                    } else {
                        max_price >= get_price(&data[i]) * 1.001
                    }
                });
                let is_min_significant = (0..data.len()).all(|i| {
                    if i == min_idx || i < start || i >= start + period {
                        true
                    } else {
                        min_price <= get_min_price(&data[i]) * 0.999
                    }
                });

                if is_max_significant && max_idx < data.len() - 1 && group_maxima.last().map_or(true, |(last_idx, _)| max_idx - last_idx >= min_pivot_distance) {
                    group_maxima.push((max_idx, max_price));
                }
                if is_min_significant && min_idx < data.len() - 1 && group_minima.last().map_or(true, |(last_idx, _)| min_idx - last_idx >= min_pivot_distance) {
                    group_minima.push((min_idx, min_price));
                }
            }
            start += 1; // Slide by 1 for overlapping windows
        }

        // Staircase (Grouping)
        for i in 0..group_maxima.len().saturating_sub(3) {
            let x0 = group_maxima[i];
            let m1 = group_minima.get(i + 1).copied().unwrap_or((0, 0.0));
            let x1 = group_maxima.get(i + 2).copied().unwrap_or((0, 0.0));
            let m2 = group_minima.get(i + 3).copied().unwrap_or((0, 0.0));
            if x0.0 < data.len() - 1 && m1.0 < data.len() - 1 && x1.0 < data.len() - 1 && m2.0 < data.len() - 1
                && x0.0 < m1.0 && m1.0 < x1.0 && x1.0 < m2.0
                && (x0.1 - m1.1) / x0.1 >= 0.001
                && (x1.1 - m1.1) / m1.1 >= 0.001
                && is_within_range(x0.1, m2.1, 0.003)
            {
                let labels = vec![
                    format!("Staircase_X0_P{}_{} ", pattern_id, period),
                    format!("Staircase_M1_P{}_{} ", pattern_id, period),
                    format!("Staircase_X1_P{}_{} ", pattern_id, period),
                    format!("Staircase_M2_P{}_{} ", pattern_id, period),
                ];
                let indices = vec![x0.0, m1.0, x1.0, m2.0];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x0.0].date, "Staircase_X0 ", x0.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "Staircase_M1 ", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "Staircase_X1 ", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "Staircase_M2 ", m2.1));
                pattern_id += 1;
            }
        }

        // Bullish Flag (Grouping)
        for i in 0..group_maxima.len().saturating_sub(5) {
            let x0 = group_maxima[i];
            let x1 = group_maxima[i + 1];
            let m1 = group_minima.get(i + 2).copied().unwrap_or((0, 0.0));
            let x2 = group_maxima.get(i + 3).copied().unwrap_or((0, 0.0));
            let m2 = group_minima.get(i + 4).copied().unwrap_or((0, 0.0));
            let x3 = group_maxima.get(i + 5).copied().unwrap_or((0, 0.0));
            if x0.0 < data.len() - 1 && x1.0 < data.len() - 1 && m1.0 < data.len() - 1 && x2.0 < data.len() - 1 && m2.0 < data.len() - 1 && x3.0 < data.len() - 1
                && x0.0 < x1.0 && x1.0 < m1.0 && m1.0 < x2.0 && x2.0 < m2.0 && m2.0 < x3.0
            {
                let slope_m1_m2 = (m2.1 - m1.1) / (m2.0 as f64 - m1.0 as f64);
                let slope_x2_x3 = (x3.1 - x2.1) / (x3.0 as f64 - x2.0 as f64);
                let flagpole_pct = (x1.1 - x0.1) / x0.1;
                if flagpole_pct >= 0.005
                    && x2.1 <= x1.1 * 1.005
                    && m2.1 >= m1.1 * 0.995
                    && x3.1 <= x1.1 * 1.005
                    && slope_m1_m2 >= -0.016 && slope_m1_m2 <= 0.016
                    && slope_x2_x3 >= -0.016 && slope_x2_x3 <= 0.016
                    && slopes_are_parallel(m1.0, m1.1, m2.0, m2.1, x2.0, x2.1, x3.0, x3.1, 0.008)
                {
                    let labels = vec![
                        format!("BullishFlag_X0_P{}_{} ", pattern_id, period),
                        format!("BullishFlag_X1_P{}_{} ", pattern_id, period),
                        format!("BullishFlag_M1_P{}_{} ", pattern_id, period),
                        format!("BullishFlag_X2_P{}_{} ", pattern_id, period),
                        format!("BullishFlag_M2_P{}_{} ", pattern_id, period),
                        format!("BullishFlag_X3_P{}_{} ", pattern_id, period),
                    ];
                    let indices = vec![x0.0, x1.0, m1.0, x2.0, m2.0, x3.0];
                    for (idx, label) in indices.iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.clone());
                    }
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x0.0].date, "BullishFlag_X0 ", x0.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "BullishFlag_X1 ", x1.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "BullishFlag_M1 ", m1.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "BullishFlag_X2 ", x2.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "BullishFlag_M2 ", m2.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "BullishFlag_X3 ", x3.1));
                    pattern_id += 1;
                }
            }
        }

        // Bearish Flag (Grouping)
        for i in 0..group_minima.len().saturating_sub(5) {
            let m0 = group_minima[i];
            let m1 = group_minima[i + 1];
            let x1 = group_maxima.get(i + 2).copied().unwrap_or((0, 0.0));
            let m2 = group_minima.get(i + 3).copied().unwrap_or((0, 0.0));
            let x2 = group_maxima.get(i + 4).copied().unwrap_or((0, 0.0));
            let m3 = group_minima.get(i + 5).copied().unwrap_or((0, 0.0));
            if m0.0 < data.len() - 1 && m1.0 < data.len() - 1 && x1.0 < data.len() - 1 && m2.0 < data.len() - 1 && x2.0 < data.len() - 1 && m3.0 < data.len() - 1
                && m0.0 < m1.0 && m1.0 < x1.0 && x1.0 < m2.0 && m2.0 < x2.0 && x2.0 < m3.0
            {
                let slope_x1_x2 = (x2.1 - x1.1) / (x2.0 as f64 - x1.0 as f64);
                let slope_m2_m3 = (m3.1 - m2.1) / (m3.0 as f64 - m2.0 as f64);
                let flagpole_pct = (m0.1 - m1.1) / m0.1;
                if flagpole_pct >= 0.005
                    && m2.1 <= m1.1 * 1.005
                    && x2.1 <= x1.1 * 1.005
                    && m3.1 <= m1.1 * 1.005
                    && slope_x1_x2 >= -0.016 && slope_x1_x2 <= 0.016
                    && slope_m2_m3 >= -0.016 && slope_m2_m3 <= 0.016
                    && slopes_are_parallel(x1.0, x1.1, x2.0, x2.1, m2.0, m2.1, m3.0, m3.1, 0.008)
                {
                    let labels = vec![
                        format!("BearishFlag_M0_P{}_{} ", pattern_id, period),
                        format!("BearishFlag_M1_P{}_{} ", pattern_id, period),
                        format!("BearishFlag_X1_P{}_{} ", pattern_id, period),
                        format!("BearishFlag_M2_P{}_{} ", pattern_id, period),
                        format!("BearishFlag_X2_P{}_{} ", pattern_id, period),
                        format!("BearishFlag_M3_P{}_{} ", pattern_id, period),
                    ];
                    let indices = vec![m0.0, m1.0, x1.0, m2.0, x2.0, m3.0];
                    for (idx, label) in indices.iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.clone());
                    }
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m0.0].date, "BearishFlag_M0 ", m0.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "BearishFlag_M1 ", m1.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "BearishFlag_X1 ", x1.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "BearishFlag_M2 ", m2.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "BearishFlag_X2 ", x2.1));
                    log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "BearishFlag_M3 ", m3.1));
                    pattern_id += 1;
                }
            }
        }

        // Rising Tops (Grouping)
        for i in 0..group_maxima.len().saturating_sub(2) {
            let x1 = group_maxima[i];
            let x2 = group_maxima[i + 1];
            let x3 = group_maxima[i + 2];
            if (x2.1 - x1.1) / x1.1 >= 0.001 && (x3.1 - x2.1) / x2.1 >= 0.001 {
                let labels = vec![
                    format!("RisingTops_X1_P{}_{} ", pattern_id, period),
                    format!("RisingTops_X2_P{}_{} ", pattern_id, period),
                    format!("RisingTops_X3_P{}_{} ", pattern_id, period),
                ];
                let indices = vec![x1.0, x2.0, x3.0];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "RisingTops_X1 ", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "RisingTops_X2 ", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "RisingTops_X3 ", x3.1));
                pattern_id += 1;
            }
        }

        // Rising Bottoms (Grouping)
        for i in 0..group_minima.len().saturating_sub(2) {
            let m1 = group_minima[i];
            let m2 = group_minima[i + 1];
            let m3 = group_minima[i + 2];
            if (m2.1 - m1.1) / m1.1 >= 0.001 && (m3.1 - m2.1) / m2.1 >= 0.001 {
                let labels = vec![
                    format!("RisingBottoms_M1_P{}_{} ", pattern_id, period),
                    format!("RisingBottoms_M2_P{}_{} ", pattern_id, period),
                    format!("RisingBottoms_M3_P{}_{} ", pattern_id, period),
                ];
                let indices = vec![m1.0, m2.0, m3.0];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "RisingBottoms_M1 ", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "RisingBottoms_M2 ", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "RisingBottoms_M3 ", m3.1));
                pattern_id += 1;
            }
        }

        // Falling Tops (Grouping)
        for i in 0..group_maxima.len().saturating_sub(2) {
            let x1 = group_maxima[i];
            let x2 = group_maxima[i + 1];
            let x3 = group_maxima[i + 2];
            if (x1.1 - x2.1) / x1.1 >= 0.001 && (x2.1 - x3.1) / x2.1 >= 0.001 {
                let labels = vec![
                    format!("FallingTops_X1_P{}_{} ", pattern_id, period),
                    format!("FallingTops_X2_P{}_{} ", pattern_id, period),
                    format!("FallingTops_X3_P{}_{} ", pattern_id, period),
                ];
                let indices = vec![x1.0, x2.0, x3.0];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "FallingTops_X1 ", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "FallingTops_X2 ", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "FallingTops_X3 ", x3.1));
                pattern_id += 1;
            }
        }

        // Falling Bottoms (Grouping)
        for i in 0..minima.len().saturating_sub(2) {
            let m1 = group_minima[i];
            let m2 = group_minima[i + 1];
            let m3 = group_minima[i + 2];
            if (m1.1 - m2.1) / m1.1 >= 0.001 && (m2.1 - m3.1) / m2.1 >= 0.001 {
                let labels = vec![
                    format!("FallingBottoms_M1_P{}_{} ", pattern_id, period),
                    format!("FallingBottoms_M2_P{}_{} ", pattern_id, period),
                    format!("FallingBottoms_M3_P{}_{} ", pattern_id, period),
                ];
                let indices = vec![m1.0, m2.0, m3.0];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "FallingBottoms_M1 ", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "FallingBottoms_M2 ", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "FallingBottoms_M3 ", m3.1));
                pattern_id += 1;
            }
        }
    }

    let mut log_file = File::create("../chart_patterns_log.csv").expect("Failed to create log file");
    writeln!(log_file, "pattern_id,date,component,price").expect("Failed to write log header");
    for entry in log_entries {
        writeln!(log_file, "{}", entry).expect("Failed to write log line");
    }
}

fn main() {
    let mut data: Vec<OHLC> = Vec::new();
    let file = File::open("../GBPAUD_daily.csv").expect("Failed to open GBPAUD_daily.csv");
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).expect("Failed to read GBPAUD_daily.csv");

    for line in contents.lines().skip(1) {
        if let Some(ohlc) = parse_ohlc(line) {
            data.push(ohlc);
        }
    }

    data.sort_by(|a, b| a.date.cmp(&b.date));
    analyze_patterns(&mut data);

    let mut file = File::create("../chart_patterns_GBPAUD_daily.csv").expect("Failed to create file");
    writeln!(file, "date,open,high,low,close,pattern").expect("Failed to write header");
    for entry in data {
        let pattern = if entry.pattern.is_empty() {
            "".to_string()
        } else {
            let mut patterns: Vec<String> = entry.pattern.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };
        writeln!(
            file,
            "{},{},{},{},{},{}",
            entry.date, entry.open, entry.high, entry.low, entry.close, pattern
        )
        .expect("Failed to write line");
    }
}