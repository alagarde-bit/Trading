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

fn analyze_head_and_shoulders(data: &mut Vec<OHLC>, timeframe: &str) {
    let mut maxima: Vec<(usize, f64)> = Vec::new();
    let mut minima: Vec<(usize, f64)> = Vec::new();
    let mut log_entries = Vec::new();
    let mut pattern_id = 1;

    // Set grouping periods based on timeframe
    let period_groups = match timeframe {
        "weekly" => vec![2, 4, 8, 12, 26, 52],  // 2 weeks to 1 year
        "monthly" => vec![2, 3, 4, 6, 12],      // 2 months to 1 year
        _ => vec![10, 20, 30, 60, 90, 183, 366], // Default daily periods
    };

    // Nearest Neighbor Pivot Detection (5-bar window)
    for i in 0..data.len() {
        let curr_price = get_price(&data[i]);
        let curr_min_price = get_min_price(&data[i]);

        let mut is_max = true;
        let mut is_min = true;
        for j in i.saturating_sub(2)..=i + 2 {
            if j < data.len() && j != i {
                if get_price(&data[j]) > curr_price {
                    is_max = false;
                }
                if get_min_price(&data[j]) < curr_min_price {
                    is_min = false;
                }
            }
        }

        if is_max {
            maxima.push((i, curr_price));
        }
        if is_min {
            minima.push((i, curr_min_price));
        }
    }

    // Head and Shoulders (Nearest Neighbor)
    for i in 0..maxima.len() {
        for j in i + 1..maxima.len() {
            for k in j + 1..maxima.len() {
                let left_shoulder = maxima[i];
                let head = maxima[j];
                let right_shoulder = maxima[k];

                if right_shoulder.0 - left_shoulder.0 > 60 {
                    continue;
                }

                if head.1 > left_shoulder.1
                    && head.1 > right_shoulder.1
                    && is_within_range(left_shoulder.1, right_shoulder.1, 0.01)
                {
                    let has_bigger_peak_ls_to_head = maxima
                        .iter()
                        .any(|&(idx, price)| idx > left_shoulder.0 && idx < head.0 && price >= head.1);
                    let has_bigger_peak_head_to_rs = maxima
                        .iter()
                        .any(|&(idx, price)| idx > head.0 && idx < right_shoulder.0 && price >= head.1);

                    if has_bigger_peak_ls_to_head || has_bigger_peak_head_to_rs {
                        continue;
                    }

                    let min_between_ls_and_head = minima
                        .iter()
                        .filter(|&&(idx, _)| idx > left_shoulder.0 && idx < head.0)
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let min_between_head_and_rs = minima
                        .iter()
                        .filter(|&&(idx, _)| idx > head.0 && idx < right_shoulder.0)
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    if let (Some(&valley1), Some(&valley2)) = (min_between_ls_and_head, min_between_head_and_rs) {
                        if valley1.1 < left_shoulder.1
                            && valley1.1 < right_shoulder.1
                            && valley2.1 < left_shoulder.1
                            && valley2.1 < right_shoulder.1
                        {
                            let ls_label = format!("HeadAndShoulders_LeftShoulder_P{}_NN", pattern_id);
                            let lv_label = format!("HeadAndShoulders_LeftValley_P{}_NN", pattern_id);
                            let h_label = format!("HeadAndShoulders_Head_P{}_NN", pattern_id);
                            let rv_label = format!("HeadAndShoulders_RightValley_P{}_NN", pattern_id);
                            let rs_label = format!("HeadAndShoulders_RightShoulder_P{}_NN", pattern_id);

                            data[left_shoulder.0].pattern.insert(ls_label.clone());
                            data[valley1.0].pattern.insert(lv_label.clone());
                            data[head.0].pattern.insert(h_label.clone());
                            data[valley2.0].pattern.insert(rv_label.clone());
                            data[right_shoulder.0].pattern.insert(rs_label.clone());

                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[left_shoulder.0].date, "LeftShoulder", left_shoulder.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[valley1.0].date, "LeftValley", valley1.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[head.0].date, "Head", head.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[valley2.0].date, "RightValley", valley2.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[right_shoulder.0].date, "RightShoulder", right_shoulder.1
                            ));

                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Inverse Head and Shoulders (Nearest Neighbor)
    for i in 0..minima.len() {
        for j in i + 1..minima.len() {
            for k in j + 1..minima.len() {
                let left_shoulder = minima[i];
                let head = minima[j];
                let right_shoulder = minima[k];

                if right_shoulder.0 - left_shoulder.0 > 60 {
                    continue;
                }

                if head.1 < left_shoulder.1
                    && head.1 < right_shoulder.1
                    && is_within_range(left_shoulder.1, right_shoulder.1, 0.01)
                {
                    let has_lower_trough_ls_to_head = minima
                        .iter()
                        .any(|&(idx, price)| idx > left_shoulder.0 && idx < head.0 && price <= head.1);
                    let has_lower_trough_head_to_rs = minima
                        .iter()
                        .any(|&(idx, price)| idx > head.0 && idx < right_shoulder.0 && price <= head.1);

                    if has_lower_trough_ls_to_head || has_lower_trough_head_to_rs {
                        continue;
                    }

                    let max_between_ls_and_head = maxima
                        .iter()
                        .filter(|&&(idx, _)| idx > left_shoulder.0 && idx < head.0)
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let max_between_head_and_rs = maxima
                        .iter()
                        .filter(|&&(idx, _)| idx > head.0 && idx < right_shoulder.0)
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    if let (Some(&peak1), Some(&peak2)) = (max_between_ls_and_head, max_between_head_and_rs) {
                        if peak1.1 > left_shoulder.1
                            && peak1.1 > right_shoulder.1
                            && peak2.1 > left_shoulder.1
                            && peak2.1 > right_shoulder.1
                        {
                            let ls_label = format!("InverseHeadAndShoulders_LeftShoulder_P{}_NN", pattern_id);
                            let lp_label = format!("InverseHeadAndShoulders_LeftPeak_P{}_NN", pattern_id);
                            let h_label = format!("InverseHeadAndShoulders_Head_P{}_NN", pattern_id);
                            let rp_label = format!("InverseHeadAndShoulders_RightPeak_P{}_NN", pattern_id);
                            let rs_label = format!("InverseHeadAndShoulders_RightShoulder_P{}_NN", pattern_id);

                            data[left_shoulder.0].pattern.insert(ls_label.clone());
                            data[peak1.0].pattern.insert(lp_label.clone());
                            data[head.0].pattern.insert(h_label.clone());
                            data[peak2.0].pattern.insert(rp_label.clone());
                            data[right_shoulder.0].pattern.insert(rs_label.clone());

                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[left_shoulder.0].date, "InverseLeftShoulder", left_shoulder.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[peak1.0].date, "InverseLeftPeak", peak1.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[head.0].date, "InverseHead", head.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[peak2.0].date, "InverseRightPeak", peak2.1
                            ));
                            log_entries.push(format!(
                                "{},{},{},{}",
                                pattern_id, data[right_shoulder.0].date, "InverseRightShoulder", right_shoulder.1
                            ));

                            pattern_id += 1;
                        }
                    }
                }
            }
        }
    }

    // Grouping Approach for Head and Shoulders and Inverse Head and Shoulders
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

        // Head and Shoulders (Grouping)
        for i in 0..group_maxima.len().saturating_sub(4) {
            let left_shoulder = group_maxima[i];
            let left_valley = group_minima[i + 1];
            let head = group_maxima[i + 2];
            let right_valley = group_minima[i + 3];
            let right_shoulder = group_maxima[i + 4];

            if right_shoulder.0 - left_shoulder.0 > 60 {
                continue;
            }

            if head.1 > left_shoulder.1
                && head.1 > right_shoulder.1
                && is_within_range(left_shoulder.1, right_shoulder.1, 0.01)
                && left_valley.1 < left_shoulder.1
                && left_valley.1 < right_shoulder.1
                && right_valley.1 < left_shoulder.1
                && right_valley.1 < right_shoulder.1
            {
                let has_bigger_peak_ls_to_head = group_maxima
                    .iter()
                    .skip(i + 1)
                    .take(i + 2 - (i + 1))
                    .any(|&(idx, price)| idx > left_shoulder.0 && idx < head.0 && price >= head.1);

                let has_bigger_peak_head_to_rs = group_maxima
                    .iter()
                    .skip(i + 3)
                    .take(i + 4 - (i + 3))
                    .any(|&(idx, price)| idx > head.0 && idx < right_shoulder.0 && price >= head.1);

                if has_bigger_peak_ls_to_head || has_bigger_peak_head_to_rs {
                    continue;
                }

                let ls_label = format!("HeadAndShoulders_LeftShoulder_P{}_{}", pattern_id, period);
                let lv_label = format!("HeadAndShoulders_LeftValley_P{}_{}", pattern_id, period);
                let h_label = format!("HeadAndShoulders_Head_P{}_{}", pattern_id, period);
                let rv_label = format!("HeadAndShoulders_RightValley_P{}_{}", pattern_id, period);
                let rs_label = format!("HeadAndShoulders_RightShoulder_P{}_{}", pattern_id, period);

                data[left_shoulder.0].pattern.insert(ls_label.clone());
                data[left_valley.0].pattern.insert(lv_label.clone());
                data[head.0].pattern.insert(h_label.clone());
                data[right_valley.0].pattern.insert(rv_label.clone());
                data[right_shoulder.0].pattern.insert(rs_label.clone());

                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[left_shoulder.0].date, "LeftShoulder", left_shoulder.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[left_valley.0].date, "LeftValley", left_valley.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[head.0].date, "Head", head.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[right_valley.0].date, "RightValley", right_valley.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[right_shoulder.0].date, "RightShoulder", right_shoulder.1
                ));

                pattern_id += 1;
            }
        }

        // Inverse Head and Shoulders (Grouping)
        for i in 0..group_minima.len().saturating_sub(4) {
            let left_shoulder = group_minima[i];
            let left_peak = group_maxima[i + 1];
            let head = group_minima[i + 2];
            let right_peak = group_maxima[i + 3]; // Removed _PROMPT:
            let right_shoulder = group_minima[i + 4];

            if right_shoulder.0 - left_shoulder.0 > 60 {
                continue;
            }

            if head.1 < left_shoulder.1
                && head.1 < right_shoulder.1
                && is_within_range(left_shoulder.1, right_shoulder.1, 0.01)
                && left_peak.1 > left_shoulder.1
                && left_peak.1 > right_shoulder.1
                && right_peak.1 > left_shoulder.1
                && right_peak.1 > right_shoulder.1
            {
                let has_lower_trough_ls_to_head = group_minima
                    .iter()
                    .skip(i + 1)
                    .take(i + 2 - (i + 1))
                    .any(|&(idx, price)| idx > left_shoulder.0 && idx < head.0 && price <= head.1);

                let has_lower_trough_head_to_rs = group_minima
                    .iter()
                    .skip(i + 3)
                    .take(i + 4 - (i + 3))
                    .any(|&(idx, price)| idx > head.0 && idx < right_shoulder.0 && price <= head.1);

                if has_lower_trough_ls_to_head || has_lower_trough_head_to_rs {
                    continue;
                }

                let ls_label = format!("InverseHeadAndShoulders_LeftShoulder_P{}_{}", pattern_id, period);
                let lp_label = format!("InverseHeadAndShoulders_LeftPeak_P{}_{}", pattern_id, period);
                let h_label = format!("InverseHeadAndShoulders_Head_P{}_{}", pattern_id, period);
                let rp_label = format!("InverseHeadAndShoulders_RightPeak_P{}_{}", pattern_id, period);
                let rs_label = format!("InverseHeadAndShoulders_RightShoulder_P{}_{}", pattern_id, period);

                data[left_shoulder.0].pattern.insert(ls_label.clone());
                data[left_peak.0].pattern.insert(lp_label.clone());
                data[head.0].pattern.insert(h_label.clone());
                data[right_peak.0].pattern.insert(rp_label.clone());
                data[right_shoulder.0].pattern.insert(rs_label.clone());

                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[left_shoulder.0].date, "InverseLeftShoulder", left_shoulder.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[left_peak.0].date, "InverseLeftPeak", left_peak.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[head.0].date, "InverseHead", head.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[right_peak.0].date, "InverseRightPeak", right_peak.1
                ));
                log_entries.push(format!(
                    "{},{},{},{}",
                    pattern_id, data[right_shoulder.0].date, "InverseRightShoulder", right_shoulder.1
                ));

                pattern_id += 1;
            }
        }
    }

    let mut log_file = File::create("../head_and_shoulders_log.csv").expect("Failed to create log file");
    writeln!(log_file, "pattern_id,date,component,price").expect("Failed to write log header");
    for entry in log_entries {
        writeln!(log_file, "{}", entry).expect("Failed to write log line");
    }
}

fn process_dataset(input_file: &str, output_file: &str, timeframe: &str) {
    let mut data: Vec<OHLC> = Vec::new();
    let file = File::open(input_file).expect(&format!("Failed to open {}", input_file));
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).expect(&format!("Failed to read {}", input_file));

    for line in contents.lines().skip(1) {
        if let Some(ohlc) = parse_ohlc(line) {
            data.push(ohlc);
        }
    }

    data.sort_by(|a, b| a.date.cmp(&b.date));
    analyze_head_and_shoulders(&mut data, timeframe);

    let mut file = File::create(output_file).expect(&format!("Failed to create {}", output_file));
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

fn main() {
    // Process GBPAUD datasets
    process_dataset("../GBPAUD_daily.csv", "../head_and_shoulders_GBPAUD_daily.csv", "daily");
    process_dataset("../GBPAUD_weekly.csv", "../head_and_shoulders_GBPAUD_weekly.csv", "weekly");
    process_dataset("../GBPAUD_monthly.csv", "../head_and_shoulders_GBPAUD_monthly.csv", "monthly");
    
    // Process NZDJPY datasets
    process_dataset("../NZDJPY_daily.csv", "../head_and_shoulders_NZDJPY_daily.csv", "daily");
    process_dataset("../NZDJPY_weekly.csv", "../head_and_shoulders_NZDJPY_weekly.csv", "weekly");
    process_dataset("../NZDJPY_monthly.csv", "../head_and_shoulders_NZDJPY_monthly.csv", "monthly");

    // Process GBPUSD datasets
    process_dataset("../GBPUSD_daily.csv", "../head_and_shoulders_GBPUSD_daily.csv", "daily");
    process_dataset("../GBPUSD_weekly.csv", "../head_and_shoulders_GBPUSD_weekly.csv", "weekly");
    process_dataset("../GBPUSD_monthly.csv", "../head_and_shoulders_GBPUSD_monthly.csv", "monthly");

    // Process USDCHF datasets
    process_dataset("../USDCHF_daily.csv", "../head_and_shoulders_USDCHF_daily.csv", "daily");
    process_dataset("../USDCHF_weekly.csv", "../head_and_shoulders_USDCHF_weekly.csv", "weekly");
    process_dataset("../USDCHF_monthly.csv", "../head_and_shoulders_USDCHF_monthly.csv", "monthly");

    // Process USDJPY datasets
    process_dataset("../USDJPY_daily.csv", "../head_and_shoulders_USDJPY_daily.csv", "daily");
    process_dataset("../USDJPY_weekly.csv", "../head_and_shoulders_USDJPY_weekly.csv", "weekly");
    process_dataset("../USDJPY_monthly.csv", "../head_and_shoulders_USDJPY_monthly.csv", "monthly");
}