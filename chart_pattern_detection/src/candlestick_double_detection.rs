use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Write};

/// Represents a candlestick with OHLC data and identified patterns.
#[derive(Debug)]
struct OHLC {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    patterns: HashSet<String>, // Stores multiple pattern labels
}

/// Parses a CSV line into an OHLC struct.
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
        patterns: HashSet::new(),
    })
}

/// Computes key candlestick metrics: total range, body, shadows, and direction.
/// Returns None if total_range is zero to avoid division errors.
fn compute_candlestick_metrics(ohlc: &OHLC) -> Option<(f64, f64, f64, f64, bool)> {
    let total_range = ohlc.high - ohlc.low;
    if total_range == 0.0 {
        return None;
    }
    let body = (ohlc.open - ohlc.close).abs();
    let upper_shadow = ohlc.high - ohlc.open.max(ohlc.close);
    let lower_shadow = ohlc.open.min(ohlc.close) - ohlc.low;
    let is_bullish = ohlc.close > ohlc.open;
    Some((total_range, body, upper_shadow, lower_shadow, is_bullish))
}

/// Checks for Classic Doji (used for Double Doji): small body, balanced shadows.
fn is_classic_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let shadow_diff = (upper_shadow - lower_shadow).abs() / total_range;
        body_ratio <= 0.15 && shadow_diff <= 0.05 // Relaxed single candlestick criteria
    } else {
        false
    }
}

/// Checks for Bullish Kicker: bearish candle followed by a gapped-up bullish candle.
fn is_bullish_kicker(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    let gap_up = bar2.open > bar1.close;
    let no_overlap = bar2.open > bar1.high;
    is_bar1_bearish && is_bar2_bullish && gap_up && no_overlap
}

/// Checks for Bearish Kicker: bullish candle followed by a gapped-down bearish candle.
fn is_bearish_kicker(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    let gap_down = bar2.open < bar1.close;
    let no_overlap = bar2.open < bar1.low;
    is_bar1_bullish && is_bar2_bearish && gap_down && no_overlap
}

fn get_threshold(timeframe: &str) -> f64 {
    match timeframe {
        "daily" => 0.003,  // 0.3%
        "weekly" => 0.01,  // 1%
        "monthly" => 0.015, // 1.5%
        _ => 0.003,        // default to daily
    }
}

/// Checks for Tweezer Bottoms: two candles with similar lows, bearish then bullish.
fn is_tweezer_bottoms(data: &[OHLC], i: usize, timeframe: &str) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    if let Some((range1, _, _, _, _)) = compute_candlestick_metrics(bar1) {
        let low_diff = (bar1.low - bar2.low).abs();
        let threshold = get_threshold(timeframe);
        let low_within_range = low_diff / range1 <= threshold;
        is_bar1_bearish && is_bar2_bullish && low_within_range
    } else {
        false
    }
}

/// Checks for Tweezer Tops: two candles with similar highs, bullish then bearish.
fn is_tweezer_tops(data: &[OHLC], i: usize, timeframe: &str) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    if let Some((range1, _, _, _, _)) = compute_candlestick_metrics(bar1) {
        let high_diff = (bar1.high - bar2.high).abs();
        let threshold = get_threshold(timeframe);
        let high_within_range = high_diff / range1 <= threshold;
        is_bar1_bullish && is_bar2_bearish && high_within_range
    } else {
        false
    }
}

/// Checks for Dark Cloud Cover: bullish candle followed by bearish candle penetrating 50%.
fn is_dark_cloud_cover(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    let gap_up = bar2.open > bar1.close;
    let penetration = bar2.close <= (bar1.open + bar1.close) / 2.0;
    is_bar1_bullish && is_bar2_bearish && gap_up && penetration
}

/// Checks for Piercing Line: bearish candle followed by bullish candle penetrating 50%.
fn is_piercing_line(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    let gap_down = bar2.open < bar1.close;
    let penetration = bar2.close >= (bar1.open + bar1.close) / 2.0;
    is_bar1_bearish && is_bar2_bullish && gap_down && penetration
}

/// Checks for Double Doji: two consecutive Doji candles.
fn is_double_doji(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    is_classic_doji(bar1) && is_classic_doji(bar2)
}

/// Checks for Matching Low: two bearish candles with similar closes.
fn is_matching_low(data: &[OHLC], i: usize, timeframe: &str) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    if let Some((range1, _, _, _, _)) = compute_candlestick_metrics(bar1) {
        let close_diff = (bar1.close - bar2.close).abs();
        let threshold = get_threshold(timeframe);
        let close_within_range = close_diff / range1 <= threshold;
        is_bar1_bearish && is_bar2_bearish && close_within_range
    } else {
        false
    }
}

/// Checks for Matching High: two bullish candles with similar closes.
fn is_matching_high(data: &[OHLC], i: usize, timeframe: &str) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    if let Some((range1, _, _, _, _)) = compute_candlestick_metrics(bar1) {
        let close_diff = (bar1.close - bar2.close).abs();
        let threshold = get_threshold(timeframe);
        let close_within_range = close_diff / range1 <= threshold;
        is_bar1_bullish && is_bar2_bullish && close_within_range
    } else {
        false
    }
}

/// Checks for Rising Window: bullish gap with no overlap.
fn is_rising_window(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    let gap_up = bar2.open > bar1.high;
    let no_overlap = bar2.low > bar1.high;
    is_bar1_bullish && is_bar2_bullish && gap_up && no_overlap
}

/// Checks for Falling Window: bearish gap with no overlap.
fn is_falling_window(data: &[OHLC], i: usize) -> bool {
    if i < 1 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 1];
    let bar2 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    let gap_down = bar2.open < bar1.low;
    let no_overlap = bar2.high < bar1.low;
    is_bar1_bearish && is_bar2_bearish && gap_down && no_overlap
}

/// Analyzes double candlestick patterns for all OHLC entries.
fn analyze_double_candlestick_patterns(data: &mut Vec<OHLC>, timeframe: &str) {
    for i in 1..data.len() {
        let bar1 = &data[i - 1];
        let bar2 = &data[i];
        if is_bullish_kicker(data, i) {
            data[i - 1].patterns.insert("Bullish_Kicker_1".to_string());
            data[i].patterns.insert("Bullish_Kicker_2".to_string());
        }
        if is_bearish_kicker(data, i) {
            data[i - 1].patterns.insert("Bearish_Kicker_1".to_string());
            data[i].patterns.insert("Bearish_Kicker_2".to_string());
        }
        if is_tweezer_bottoms(data, i, timeframe) {
            data[i - 1].patterns.insert("Tweezer_Bottoms_1".to_string());
            data[i].patterns.insert("Tweezer_Bottoms_2".to_string());
        }
        if is_tweezer_tops(data, i, timeframe) {
            data[i - 1].patterns.insert("Tweezer_Tops_1".to_string());
            data[i].patterns.insert("Tweezer_Tops_2".to_string());
        }
        if is_dark_cloud_cover(data, i) {
            data[i - 1].patterns.insert("Dark_Cloud_Cover_1".to_string());
            data[i].patterns.insert("Dark_Cloud_Cover_2".to_string());
        }
        if is_piercing_line(data, i) {
            data[i - 1].patterns.insert("Piercing_Line_1".to_string());
            data[i].patterns.insert("Piercing_Line_2".to_string());
        }
        if is_double_doji(data, i) {
            data[i - 1].patterns.insert("Double_Doji_1".to_string());
            data[i].patterns.insert("Double_Doji_2".to_string());
        }
        if is_matching_low(data, i, timeframe) {
            data[i - 1].patterns.insert("Matching_Low_1".to_string());
            data[i].patterns.insert("Matching_Low_2".to_string());
        }
        if is_matching_high(data, i, timeframe) {
            data[i - 1].patterns.insert("Matching_High_1".to_string());
            data[i].patterns.insert("Matching_High_2".to_string());
        }
        if is_rising_window(data, i) {
            data[i - 1].patterns.insert("Rising_Window_1".to_string());
            data[i].patterns.insert("Rising_Window_2".to_string());
        }
        if is_falling_window(data, i) {
            data[i - 1].patterns.insert("Falling_Window_1".to_string());
            data[i].patterns.insert("Falling_Window_2".to_string());
        }
    }
}

fn process_timeframe(input_file: &str, output_file: &str, timeframe: &str) {
    // Read input CSV
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

    // Analyze double candlestick patterns
    analyze_double_candlestick_patterns(&mut data, timeframe);

    // Write output CSV
    let mut file = File::create(output_file)
        .expect(&format!("Failed to create output file {}", output_file));
    writeln!(file, "date,open,high,low,close,patterns").expect("Failed to write header");

    for entry in data {
        let patterns = if entry.patterns.is_empty() {
            "".to_string()
        } else {
            let mut patterns: Vec<String> = entry.patterns.into_iter().collect();
            patterns.sort(); // Consistent output order
            patterns.join(";")
        };
        writeln!(
            file,
            "{},{},{},{},{},{}",
            entry.date, entry.open, entry.high, entry.low, entry.close, patterns
        )
        .expect("Failed to write line");
    }
}

fn main() {
    // Process GBPAUD datasets
    process_timeframe(
        "../GBPAUD_daily.csv",
        "../double_candlestick_GBPAUD_daily.csv",
        "daily"
    );

    process_timeframe(
        "../GBPAUD_weekly.csv",
        "../double_candlestick_GBPAUD_weekly.csv",
        "weekly"
    );

    process_timeframe(
        "../GBPAUD_monthly.csv",
        "../double_candlestick_GBPAUD_monthly.csv",
        "monthly"
    );
    
    // Process NZDJPY datasets
    process_timeframe(
        "../NZDJPY_daily.csv",
        "../double_candlestick_NZDJPY_daily.csv",
        "daily"
    );

    process_timeframe(
        "../NZDJPY_weekly.csv",
        "../double_candlestick_NZDJPY_weekly.csv",
        "weekly"
    );

    process_timeframe(
        "../NZDJPY_monthly.csv",
        "../double_candlestick_NZDJPY_monthly.csv",
        "monthly"
    );
}