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
    pattern: HashSet<String>, // Changed to HashSet for distinct labels
}

fn parse_ohlc(line: &str) -> Option<OHLC> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() != 5 {
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

fn is_within_range(a: f64, b: f64, percentage: f64) -> bool {
    if a == 0.0 {
        return b == 0.0;
    }
    let diff = (a - b).abs();
    diff / a.abs() <= percentage
}

fn get_price(ohlc: &OHLC) -> f64 {
    ohlc.open.max(ohlc.close)
}

fn get_min_price(ohlc: &OHLC) -> f64 {
    ohlc.open.min(ohlc.close)
}

fn bodies_overlap(bar1: &OHLC, bar2: &OHLC) -> bool {
    let bar1_min = bar1.open.min(bar1.close);
    let bar1_max = bar1.open.max(bar1.close);
    let bar2_min = bar2.open.min(bar2.close);
    let bar2_max = bar2.open.max(bar2.close);

    bar1_max >= bar2_min && bar2_max >= bar1_min
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

    // Nearest Neighbor: Symmetrical Triangle
    for i in 0..maxima.len() {
        let (m1_idx, m1) = maxima[i];
        if let Some(&(x1_idx, x1)) = minima.iter().find(|&&(idx, _)| idx > m1_idx) {
            if let Some(&(m2_idx, m2)) = maxima.iter().find(|&&(idx, _)| idx > x1_idx) {
                if let Some(&(x2_idx, x2)) = minima.iter().find(|&&(idx, _)| idx > m2_idx) {
                    if let Some(&(m3_idx, m3)) = maxima.iter().find(|&&(idx, _)| idx > x2_idx) {
                        if m2 < m1 && x2 > x1 && m3 < m2 {
                            let indices = vec![m1_idx, x1_idx, m2_idx, x2_idx, m3_idx];
                            let labels = vec![
                                format!("Symmetrical_M1_P{}_NN", pattern_id),
                                format!("Symmetrical_X1_P{}_NN", pattern_id),
                                format!("Symmetrical_M2_P{}_NN", pattern_id),
                                format!("Symmetrical_X2_P{}_NN", pattern_id),
                                format!("Symmetrical_M3_P{}_NN", pattern_id),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone()); // Use insert for HashSet
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "Symmetrical_M1", m1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "Symmetrical_X1", x1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "Symmetrical_M2", m2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "Symmetrical_X2", x2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m3_idx].date, "Symmetrical_M3", m3));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Nearest Neighbor: Ascending Triangle
    for i in 0..maxima.len() {
        let (m1_idx, m1) = maxima[i];
        if let Some(&(x1_idx, x1)) = minima.iter().find(|&&(idx, _)| idx > m1_idx) {
            if let Some(&(m2_idx, m2)) = maxima.iter().find(|&&(idx, _)| idx > x1_idx) {
                if let Some(&(x2_idx, x2)) = minima.iter().find(|&&(idx, _)| idx > m2_idx) {
                    if let Some(&(m3_idx, m3)) = maxima.iter().find(|&&(idx, _)| idx > x2_idx) {
                        if m2 > m1 && is_within_range(x1, x2, 0.01) && m3 > m2 {
                            let indices = vec![m1_idx, x1_idx, m2_idx, x2_idx, m3_idx];
                            let labels = vec![
                                format!("Ascending_M1_P{}_NN", pattern_id),
                                format!("Ascending_X1_P{}_NN", pattern_id),
                                format!("Ascending_M2_P{}_NN", pattern_id),
                                format!("Ascending_X2_P{}_NN", pattern_id),
                                format!("Ascending_M3_P{}_NN", pattern_id),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone());
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "Ascending_M1", m1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "Ascending_X1", x1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "Ascending_M2", m2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "Ascending_X2", x2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m3_idx].date, "Ascending_M3", m3));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Nearest Neighbor: Descending Triangle
    for i in 0..maxima.len() {
        let (x1_idx, x1) = maxima[i];
        if let Some(&(m1_idx, m1)) = minima.iter().find(|&&(idx, _)| idx > x1_idx) {
            if let Some(&(x2_idx, x2)) = maxima.iter().find(|&&(idx, _)| idx > m1_idx) {
                if let Some(&(m2_idx, m2)) = minima.iter().find(|&&(idx, _)| idx > x2_idx) {
                    if let Some(&(x3_idx, x3)) = maxima.iter().find(|&&(idx, _)| idx > m2_idx) {
                        if x2 < x1 && is_within_range(m1, m2, 0.01) && x3 < x2 {
                            let indices = vec![x1_idx, m1_idx, x2_idx, m2_idx, x3_idx];
                            let labels = vec![
                                format!("Descending_X1_P{}_NN", pattern_id),
                                format!("Descending_M1_P{}_NN", pattern_id),
                                format!("Descending_X2_P{}_NN", pattern_id),
                                format!("Descending_M2_P{}_NN", pattern_id),
                                format!("Descending_X3_P{}_NN", pattern_id),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone());
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "Descending_X1", x1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "Descending_M1", m1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "Descending_X2", x2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "Descending_M2", m2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x3_idx].date, "Descending_X3", x3));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Grouping Approach for Triangle Patterns
    let period_groups = match timeframe {
        "daily" => vec![10, 20, 30, 60, 90, 183, 366],
        "weekly" => vec![2, 4, 8, 12, 26, 52],
        "monthly" => vec![2, 3, 4, 6, 12],
        _ => vec![10, 20, 30, 60, 90, 183, 366], // default to daily
    };

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

        // Symmetrical Triangle (Grouping)
        for i in 0..group_maxima.len().saturating_sub(4) {
            let m1 = group_maxima[i];
            let x1 = group_minima[i + 1];
            let m2 = group_maxima[i + 2];
            let x2 = group_minima[i + 3];
            let m3 = group_maxima[i + 4];

            if m3.0 - m1.0 > period * 5 {
                continue;
            }

            if m2.1 < m1.1 && x2.1 > x1.1 && m3.1 < m2.1 {
                let indices = vec![m1.0, x1.0, m2.0, x2.0, m3.0];
                let labels = vec![
                    format!("Symmetrical_M1_P{}_{}", pattern_id, period),
                    format!("Symmetrical_X1_P{}_{}", pattern_id, period),
                    format!("Symmetrical_M2_P{}_{}", pattern_id, period),
                    format!("Symmetrical_X2_P{}_{}", pattern_id, period),
                    format!("Symmetrical_M3_P{}_{}", pattern_id, period),
                ];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "Symmetrical_M1", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "Symmetrical_X1", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "Symmetrical_M2", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "Symmetrical_X2", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "Symmetrical_M3", m3.1));
                pattern_id += 1;
            }
        }

        // Ascending Triangle (Grouping)
        for i in 0..group_maxima.len().saturating_sub(4) {
            let m1 = group_maxima[i];
            let x1 = group_minima[i + 1];
            let m2 = group_maxima[i + 2];
            let x2 = group_minima[i + 3];
            let m3 = group_maxima[i + 4];

            if m3.0 - m1.0 > period * 5 {
                continue;
            }

            if m2.1 > m1.1 && is_within_range(x1.1, x2.1, 0.01) && m3.1 > m2.1 {
                let indices = vec![m1.0, x1.0, m2.0, x2.0, m3.0];
                let labels = vec![
                    format!("Ascending_M1_P{}_{}", pattern_id, period),
                    format!("Ascending_X1_P{}_{}", pattern_id, period),
                    format!("Ascending_M2_P{}_{}", pattern_id, period),
                    format!("Ascending_X2_P{}_{}", pattern_id, period),
                    format!("Ascending_M3_P{}_{}", pattern_id, period),
                ];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "Ascending_M1", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "Ascending_X1", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "Ascending_M2", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "Ascending_X2", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m3.0].date, "Ascending_M3", m3.1));
                pattern_id += 1;
            }
        }

        // Descending Triangle (Grouping)
        for i in 0..group_maxima.len().saturating_sub(4) {
            let x1 = group_maxima[i];
            let m1 = group_minima[i + 1];
            let x2 = group_maxima[i + 2];
            let m2 = group_minima[i + 3];
            let x3 = group_maxima[i + 4];

            if x3.0 - x1.0 > period * 5 {
                continue;
            }

            if x2.1 < x1.1 && is_within_range(m1.1, m2.1, 0.01) && x3.1 < x2.1 {
                let indices = vec![x1.0, m1.0, x2.0, m2.0, x3.0];
                let labels = vec![
                    format!("Descending_X1_P{}_{}", pattern_id, period),
                    format!("Descending_M1_P{}_{}", pattern_id, period),
                    format!("Descending_X2_P{}_{}", pattern_id, period),
                    format!("Descending_M2_P{}_{}", pattern_id, period),
                    format!("Descending_X3_P{}_{}", pattern_id, period),
                ];
                for (idx, label) in indices.iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.clone());
                }
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x1.0].date, "Descending_X1", x1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m1.0].date, "Descending_M1", m1.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x2.0].date, "Descending_X2", x2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[m2.0].date, "Descending_M2", m2.1));
                log_entries.push(format!("{},{},{},{}", pattern_id, data[x3.0].date, "Descending_X3", x3.1));
                pattern_id += 1;
            }
        }
    }

    // Broadening Formation (Nearest Neighbor)
    for i in 0..minima.len() {
        let (m1_idx, m1) = minima[i];
        if let Some(&(x1_idx, x1)) = maxima.iter().find(|&&(idx, _)| idx > m1_idx) {
            if let Some(&(m2_idx, m2)) = minima.iter().find(|&&(idx, _)| idx > x1_idx) {
                if let Some(&(x2_idx, x2)) = maxima.iter().find(|&&(idx, _)| idx > m2_idx) {
                    if let Some(&(m3_idx, m3)) = minima.iter().find(|&&(idx, _)| idx > x2_idx) {
                        if m2 < m1 && x2 > x1 && m3 < m2 {
                            let indices = vec![m1_idx, x1_idx, m2_idx, x2_idx, m3_idx];
                            let labels = vec![
                                format!("Broadening_M1_P{}_NN", pattern_id),
                                format!("Broadening_X1_P{}_NN", pattern_id),
                                format!("Broadening_M2_P{}_NN", pattern_id),
                                format!("Broadening_X2_P{}_NN", pattern_id),
                                format!("Broadening_M3_P{}_NN", pattern_id),
                            ];
                            for (idx, label) in indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.clone());
                            }
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m1_idx].date, "Broadening_M1", m1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x1_idx].date, "Broadening_X1", x1));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m2_idx].date, "Broadening_M2", m2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[x2_idx].date, "Broadening_X2", x2));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[m3_idx].date, "Broadening_M3", m3));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Tower Top (Nearest Neighbor)
    for i in 0..data.len() {
        if data[i].close > data[i].open {
            let open_i = data[i].open;
            if open_i == 0.0 {
                continue;
            }
            let span1 = (data[i].close - data[i].open).abs() / open_i;
            for bars_between in 1..=5 {
                let j = i + bars_between + 1;
                if j >= data.len() {
                    break;
                }
                if data[j].open > data[j].close {
                    let open_j = data[j].open;
                    if open_j == 0.0 {
                        continue;
                    }
                    let span2 = (data[j].open - data[j].close).abs() / open_j;
                    if bodies_overlap(&data[i], &data[j]) {
                        let mut valid = true;
                        for k in i + 1..j {
                            let open_k = data[k].open;
                            if open_k == 0.0 {
                                valid = false;
                                break;
                            }
                            let span_between = (data[k].close - data[k].open).abs() / open_k;
                            if span_between >= span1.min(span2) {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            let labels = vec![
                                format!("TowerTop_Bull1_P{}_NN", pattern_id),
                                format!("TowerTop_Bear2_P{}_NN", pattern_id),
                            ];
                            data[i].pattern.insert(labels[0].clone());
                            data[j].pattern.insert(labels[1].clone());
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[i].date, "TowerTop_Bull1", get_price(&data[i])));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[j].date, "TowerTop_Bear2", get_price(&data[j])));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Tower Bottom (Nearest Neighbor)
    for i in 0..data.len() {
        if data[i].open > data[i].close {
            let open_i = data[i].open;
            if open_i == 0.0 {
                continue;
            }
            let span1 = (data[i].open - data[i].close).abs() / open_i;
            for bars_between in 1..=5 {
                let j = i + bars_between + 1;
                if j >= data.len() {
                    break;
                }
                if data[j].close > data[j].open {
                    let open_j = data[j].open;
                    if open_j == 0.0 {
                        continue;
                    }
                    let span2 = (data[j].close - data[j].open).abs() / open_j;
                    if bodies_overlap(&data[i], &data[j]) {
                        let mut valid = true;
                        for k in i + 1..j {
                            let open_k = data[k].open;
                            if open_k == 0.0 {
                                valid = false;
                                break;
                            }
                            let span_between = (data[k].close - data[k].open).abs() / open_k;
                            if span_between >= span1.min(span2) {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            let labels = vec![
                                format!("TowerBottom_Bear1_P{}_NN", pattern_id),
                                format!("TowerBottom_Bull2_P{}_NN", pattern_id),
                            ];
                            data[i].pattern.insert(labels[0].clone());
                            data[j].pattern.insert(labels[1].clone());
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[i].date, "TowerBottom_Bear1", get_min_price(&data[i])));
                            log_entries.push(format!("{},{},{},{}", pattern_id, data[j].date, "TowerBottom_Bull2", get_price(&data[j])));
                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    let mut log_file = match File::create("../triangle_patterns_log.csv") {
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

fn process_dataset(input_file: &str, output_file: &str, timeframe: &str) {
    let mut data: Vec<OHLC> = Vec::new();
    let file = match File::open(input_file) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening input file '{}': {}", input_file, e);
            std::process::exit(1);
        }
    };

    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    if let Err(e) = reader.read_to_string(&mut contents) {
        eprintln!("Error reading input file '{}': {}", input_file, e);
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
    println!("Starting pattern analysis on {} data points for {}...", data.len(), input_file);
    analyze_patterns(&mut data, timeframe);
    println!("Pattern analysis complete for {}.", input_file);

    let mut file = match File::create(output_file) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error creating output file '{}': {}", output_file, e);
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
            patterns.sort(); // Sort for consistent output
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
        lines_written, output_file
    );
}

fn main() {
    // Process GBPAUD datasets
    process_dataset(
        "../GBPAUD_daily.csv",
        "../intermediate2_GBPAUD_daily.csv",
        "daily"
    );

    process_dataset(
        "../GBPAUD_weekly.csv",
        "../intermediate2_GBPAUD_weekly.csv",
        "weekly"
    );

    process_dataset(
        "../GBPAUD_monthly.csv",
        "../intermediate2_GBPAUD_monthly.csv",
        "monthly"
    );
    
    // Process NZDJPY datasets
    process_dataset(
        "../NZDJPY_daily.csv",
        "../intermediate2_NZDJPY_daily.csv",
        "daily"
    );

    process_dataset(
        "../NZDJPY_weekly.csv",
        "../intermediate2_NZDJPY_weekly.csv",
        "weekly"
    );

    process_dataset(
        "../NZDJPY_monthly.csv",
        "../intermediate2_NZDJPY_monthly.csv",
        "monthly"
    );
}