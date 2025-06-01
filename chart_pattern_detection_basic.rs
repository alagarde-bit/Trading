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
    diff / a <= percentage
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

fn analyze_patterns(data: &mut Vec<OHLC>) {
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    let mut minima: Vec<(usize, f64)> = Vec::new();

    for i in 0..data.len() {
        let curr_price = get_price(&data[i]);
        let curr_min_price = get_min_price(&data[i]);

        let is_max = (i == 0 || curr_price >= get_price(&data[i - 1]))
            && (i == data.len() - 1 || curr_price >= get_price(&data[i + 1]));
        let is_min = (i == 0 || curr_min_price <= get_min_price(&data[i - 1]))
            && (i == data.len() - 1 || curr_min_price <= get_min_price(&data[i + 1]));

        if is_max {
            maxima.push((i, curr_price));
        }
        if is_min {
            minima.push((i, curr_min_price));
        }
    }

    // Nearest Neighbor Patterns
    // Rectangle Pattern (Min-Max-Min-Max or Max-Min-Max-Min)
    for i in 1..maxima.len() {
        for j in 1..minima.len() {
            let (max1_idx, max1) = maxima[i - 1];
            let (max2_idx, max2) = maxima[i];
            let (min1_idx, min1) = minima[j - 1];
            let (min2_idx, min2) = minima[j];

            if is_within_range(max1, max2, 0.002) && is_within_range(min1, min2, 0.002) {
                let indices = vec![max1_idx, max2_idx, min1_idx, min2_idx];
                let unique_indices = indices.iter().collect::<HashSet<_>>();

                if unique_indices.len() == 4 {
                    let mut sorted_indices = indices.clone();
                    sorted_indices.sort();

                    let is_min_max_min_max = minima.contains(&(sorted_indices[0], get_min_price(&data[sorted_indices[0]])))
                        && maxima.contains(&(sorted_indices[1], get_price(&data[sorted_indices[1]])))
                        && minima.contains(&(sorted_indices[2], get_min_price(&data[sorted_indices[2]])))
                        && maxima.contains(&(sorted_indices[3], get_price(&data[sorted_indices[3]])));
                    let is_max_min_max_min = maxima.contains(&(sorted_indices[0], get_price(&data[sorted_indices[0]])))
                        && minima.contains(&(sorted_indices[1], get_min_price(&data[sorted_indices[1]])))
                        && maxima.contains(&(sorted_indices[2], get_price(&data[sorted_indices[2]])))
                        && minima.contains(&(sorted_indices[3], get_min_price(&data[sorted_indices[3]])));

                    if is_min_max_min_max {
                        let labels = vec!["Rectangle_Min1_NN", "Rectangle_Max2_NN", "Rectangle_Min3_NN", "Rectangle_Max4_NN"];
                        for (idx, label) in sorted_indices.iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.to_string());
                        }
                    } else if is_max_min_max_min {
                        let labels = vec!["Rectangle_Max1_NN", "Rectangle_Min2_NN", "Rectangle_Max3_NN", "Rectangle_Min4_NN"];
                        for (idx, label) in sorted_indices.iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.to_string());
                        }
                    }
                }
            }
        }
    }

    // Pipe Top: First bar green, second bar red
    for i in 0..data.len() - 1 {
        let bar1 = &data[i];
        let bar2 = &data[i + 1];
        let gap1 = (bar1.close - bar1.open) / bar1.open;
        let gap2 = (bar2.open - bar2.close) / bar2.open;

        if gap1 >= 0.003
            && gap2 >= 0.003
            && bar1.close > bar1.open
            && bar2.open > bar2.close
            && is_within_range(bar1.close, bar2.open, 0.003)
        {
            let prior_close = if i > 0 { data[i - 1].close } else { f64::INFINITY };
            let next_close = if i + 2 < data.len() { data[i + 2].close } else { f64::INFINITY };
            if prior_close < bar1.close && next_close < bar2.close {
                let labels = vec!["PipeTop_Min1", "PipeTop_Min2"];
                for (idx, label) in [i, i + 1].iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.to_string());
                }
            }
        }
    }

    // Pipe Bottom: First bar red, second bar green
    for i in 0..data.len() - 1 {
        let bar1 = &data[i];
        let bar2 = &data[i + 1];
        let gap1 = (bar1.open - bar1.close) / bar1.open;
        let gap2 = (bar2.close - bar2.open) / bar2.open;

        if gap1 >= 0.003
            && gap2 >= 0.003
            && bar1.open > bar1.close
            && bar2.close > bar2.open
            && is_within_range(bar1.close, bar2.open, 0.003)
        {
            let prior_close = if i > 0 { data[i - 1].close } else { f64::NEG_INFINITY };
            let next_close = if i + 2 < data.len() { data[i + 2].close } else { f64::NEG_INFINITY };
            if prior_close > bar1.close && next_close > bar2.close {
                let labels = vec!["PipeBottom_Max1", "PipeBottom_Max2"];
                for (idx, label) in [i, i + 1].iter().zip(labels.iter()) {
                    data[*idx].pattern.insert(label.to_string());
                }
            }
        }
    }

    // Double Top (Nearest Neighbor)
    for i in 1..maxima.len() {
        let (idx1, max1) = maxima[i - 1];
        let (idx2, max2) = maxima[i];
        if is_within_range(max1, max2, 0.002) {
            if let Some(&(trough_idx, trough)) = minima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2) {
                if (max1 - trough) / max1 >= 0.005 {
                    let labels = vec!["DoubleTop_Max1_NN", "DoubleTop_Min2_NN", "DoubleTop_Max3_NN"];
                    for (idx, label) in [idx1, trough_idx, idx2].iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.to_string());
                    }
                }
            }
        }
    }

    // Double Bottom (Nearest Neighbor)
    for i in 1..minima.len() {
        let (idx1, min1) = minima[i - 1];
        let (idx2, min2) = minima[i];
        if is_within_range(min1, min2, 0.002) {
            if let Some(&(peak_idx, peak)) = maxima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2) {
                if (peak - min1) / min1 >= 0.005 {
                    let labels = vec!["DoubleBottom_Min1_NN", "DoubleBottom_Max2_NN", "DoubleBottom_Min3_NN"];
                    for (idx, label) in [idx1, peak_idx, idx2].iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.to_string());
                    }
                }
            }
        }
    }

    // Triple Top (Nearest Neighbor)
    for i in 2..maxima.len() {
        let (idx1, max1) = maxima[i - 2];
        let (idx2, max2) = maxima[i - 1];
        let (idx3, max3) = maxima[i];
        if is_within_range(max1, max2, 0.002) && is_within_range(max2, max3, 0.002) {
            let trough1 = minima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2);
            let trough2 = minima.iter().find(|&&(idx, _)| idx > idx2 && idx < idx3);
            if let (Some(&(t1_idx, t1)), Some(&(t2_idx, t2))) = (trough1, trough2) {
                if (max1 - t1) / max1 >= 0.005 && (max2 - t2) / max2 >= 0.005 {
                    let labels = vec![
                        "TripleTop_Max1_NN",
                        "TripleTop_Min2_NN",
                        "TripleTop_Max3_NN",
                        "TripleTop_Min4_NN",
                        "TripleTop_Max5_NN",
                    ];
                    for (idx, label) in [idx1, t1_idx, idx2, t2_idx, idx3].iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.to_string());
                    }
                }
            }
        }
    }

    // Triple Bottom (Nearest Neighbor)
    for i in 2..minima.len() {
        let (idx1, min1) = minima[i - 2];
        let (idx2, min2) = minima[i - 1];
        let (idx3, min3) = minima[i];
        if is_within_range(min1, min2, 0.002) && is_within_range(min2, min3, 0.002) {
            let peak1 = maxima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2);
            let peak2 = maxima.iter().find(|&&(idx, _)| idx > idx2 && idx < idx3);
            if let (Some(&(p1_idx, p1)), Some(&(p2_idx, p2))) = (peak1, peak2) {
                if (p1 - min1) / min1 >= 0.005 && (p2 - min2) / min2 >= 0.005 {
                    let labels = vec![
                        "TripleBottom_Min1_NN",
                        "TripleBottom_Max2_NN",
                        "TripleBottom_Min3_NN",
                        "TripleBottom_Max4_NN",
                        "TripleBottom_Min5_NN",
                    ];
                    for (idx, label) in [idx1, p1_idx, idx2, p2_idx, idx3].iter().zip(labels.iter()) {
                        data[*idx].pattern.insert(label.to_string());
                    }
                }
            }
        }
    }

    // Grouping Approach for Rectangle, Double Top/Bottom, Triple Top/Bottom
    let period_groups = vec![10, 20, 30, 60, 90, 183, 366];
    for &period in &period_groups {
        let mut group_maxima: Vec<(usize, f64)> = Vec::new();
        let mut group_minima: Vec<(usize, f64)> = Vec::new();

        let mut start = 0;
        while start < data.len() {
            let end = (start + period).min(data.len());
            if let Some(((max_idx, max_price), (min_idx, min_price))) = compute_group_pivots(data, start, end) {
                group_maxima.push((max_idx, max_price));
                group_minima.push((min_idx, min_price));
            }
            start += period;
        }

        // Rectangle (Grouping)
        for i in 1..group_maxima.len() {
            for j in 1..group_minima.len() {
                let (max1_idx, max1) = group_maxima[i - 1];
                let (max2_idx, max2) = group_maxima[i];
                let (min1_idx, min1) = group_minima[j - 1];
                let (min2_idx, min2) = group_minima[j];

                if is_within_range(max1, max2, 0.002) && is_within_range(min1, min2, 0.002) {
                    let indices = vec![max1_idx, max2_idx, min1_idx, min2_idx];
                    let unique_indices = indices.iter().collect::<HashSet<_>>();

                    if unique_indices.len() == 4 {
                        let mut sorted_indices = indices.clone();
                        sorted_indices.sort();

                        let is_min_max_min_max = group_minima.contains(&(sorted_indices[0], get_min_price(&data[sorted_indices[0]])))
                            && group_maxima.contains(&(sorted_indices[1], get_price(&data[sorted_indices[1]])))
                            && group_minima.contains(&(sorted_indices[2], get_min_price(&data[sorted_indices[2]])))
                            && group_maxima.contains(&(sorted_indices[3], get_price(&data[sorted_indices[3]])));
                        let is_max_min_max_min = group_maxima.contains(&(sorted_indices[0], get_price(&data[sorted_indices[0]])))
                            && group_minima.contains(&(sorted_indices[1], get_min_price(&data[sorted_indices[1]])))
                            && group_maxima.contains(&(sorted_indices[2], get_price(&data[sorted_indices[2]])))
                            && group_minima.contains(&(sorted_indices[3], get_min_price(&data[sorted_indices[3]])));

                        if is_min_max_min_max {
                            let labels = vec![
                                format!("Rectangle_Min1_{}", period),
                                format!("Rectangle_Max2_{}", period),
                                format!("Rectangle_Min3_{}", period),
                                format!("Rectangle_Max4_{}", period),
                            ];
                            for (idx, label) in sorted_indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.to_string());
                            }
                        } else if is_max_min_max_min {
                            let labels = vec![
                                format!("Rectangle_Max1_{}", period),
                                format!("Rectangle_Min2_{}", period),
                                format!("Rectangle_Max3_{}", period),
                                format!("Rectangle_Min4_{}", period),
                            ];
                            for (idx, label) in sorted_indices.iter().zip(labels.iter()) {
                                data[*idx].pattern.insert(label.to_string());
                            }
                        }
                    }
                }
            }
        }

        // Double Top (Grouping)
        for i in 1..group_maxima.len() {
            let (idx1, max1) = group_maxima[i - 1];
            let (idx2, max2) = group_maxima[i];
            if is_within_range(max1, max2, 0.002) {
                if let Some(&(trough_idx, trough)) = minima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2) {
                    if (max1 - trough) / max1 >= 0.005 {
                        let labels = vec![
                            format!("DoubleTop_Max1_{}", period),
                            format!("DoubleTop_Min2_{}", period),
                            format!("DoubleTop_Max3_{}", period),
                        ];
                        for (idx, label) in [idx1, trough_idx, idx2].iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.to_string());
                        }
                    }
                }
            }
        }

        // Double Bottom (Grouping)
        for i in 1..group_minima.len() {
            let (idx1, min1) = group_minima[i - 1];
            let (idx2, min2) = group_minima[i];
            if is_within_range(min1, min2, 0.002) {
                if let Some(&(peak_idx, peak)) = maxima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2) {
                    if (peak - min1) / min1 >= 0.005 {
                        let labels = vec![
                            format!("DoubleBottom_Min1_{}", period),
                            format!("DoubleBottom_Max2_{}", period),
                            format!("DoubleBottom_Min3_{}", period),
                        ];
                        for (idx, label) in [idx1, peak_idx, idx2].iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.to_string());
                        }
                    }
                }
            }
        }

        // Triple Top (Grouping)
        for i in 2..group_maxima.len() {
            let (idx1, max1) = group_maxima[i - 2];
            let (idx2, max2) = group_maxima[i - 1];
            let (idx3, max3) = group_maxima[i];
            if is_within_range(max1, max2, 0.002) && is_within_range(max2, max3, 0.002) {
                let trough1 = minima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2);
                let trough2 = minima.iter().find(|&&(idx, _)| idx > idx2 && idx < idx3);
                if let (Some(&(t1_idx, t1)), Some(&(t2_idx, t2))) = (trough1, trough2) {
                    if (max1 - t1) / max1 >= 0.005 && (max2 - t2) / max2 >= 0.005 {
                        let labels = vec![
                            format!("TripleTop_Max1_{}", period),
                            format!("TripleTop_Min2_{}", period),
                            format!("TripleTop_Max3_{}", period),
                            format!("TripleTop_Min4_{}", period),
                            format!("TripleTop_Max5_{}", period),
                        ];
                        for (idx, label) in [idx1, t1_idx, idx2, t2_idx, idx3].iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.to_string());
                        }
                    }
                }
            }
        }

        // Triple Bottom (Grouping)
        for i in 2..group_minima.len() {
            let (idx1, min1) = group_minima[i - 2];
            let (idx2, min2) = group_minima[i - 1];
            let (idx3, min3) = group_minima[i];
            if is_within_range(min1, min2, 0.002) && is_within_range(min2, min3, 0.002) {
                let peak1 = maxima.iter().find(|&&(idx, _)| idx > idx1 && idx < idx2);
                let peak2 = maxima.iter().find(|&&(idx, _)| idx > idx2 && idx < idx3);
                if let (Some(&(p1_idx, p1)), Some(&(p2_idx, p2))) = (peak1, peak2) {
                    if (p1 - min1) / min1 >= 0.005 && (p2 - min2) / min2 >= 0.005 {
                        let labels = vec![
                            format!("TripleBottom_Min1_{}", period),
                            format!("TripleBottom_Max2_{}", period),
                            format!("TripleBottom_Min3_{}", period),
                            format!("TripleBottom_Max4_{}", period),
                            format!("TripleBottom_Min5_{}", period),
                        ];
                        for (idx, label) in [idx1, p1_idx, idx2, p2_idx, idx3].iter().zip(labels.iter()) {
                            data[*idx].pattern.insert(label.to_string());
                        }
                    }
                }
            }
        }
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

    let mut file = File::create("../intermediate_GBPAUD_daily.csv").expect("Failed to create file");
    writeln!(file, "date,open,high,low,close,pattern").expect("Failed to write header");
    for entry in data {
        let pattern = if entry.pattern.is_empty() {
            "".to_string()
        } else {
            let mut patterns: Vec<String> = entry.pattern.into_iter().collect();
            patterns.sort(); // Sort for consistent output
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