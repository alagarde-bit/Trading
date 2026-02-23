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

/// Checks for Classic Doji: small body, balanced shadows.
fn is_classic_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let shadow_diff = (upper_shadow - lower_shadow).abs() / total_range;
        body_ratio <= 0.15 && shadow_diff <= 0.05 // Relaxed from 0.1 and 0.02
    } else {
        false
    }
}

/// Checks for Bullish Doji: small body, longer lower shadow.
fn is_bullish_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        body_ratio <= 0.15 && upper_shadow < lower_shadow + 0.05 * total_range // Relaxed from 0.1 and 0.02
    } else {
        false
    }
}

/// Checks for Bearish Doji: small body, longer upper shadow.
fn is_bearish_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        body_ratio <= 0.15 && upper_shadow > lower_shadow - 0.05 * total_range // Relaxed from 0.1 and 0.02
    } else {
        false
    }
}

/// Checks for Dragonfly Doji: small body, negligible upper shadow, dominant lower shadow.
fn is_dragonfly_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        body_ratio <= 0.15 && upper_shadow_ratio < 0.05 && lower_shadow_ratio > 0.80 // Relaxed from 0.1, 0.02, 0.88
    } else {
        false
    }
}

/// Checks for Gravestone Doji: small body, negligible lower shadow, dominant upper shadow.
fn is_gravestone_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        body_ratio <= 0.15 && lower_shadow_ratio < 0.10 && upper_shadow_ratio > 0.75 // Relaxed from 0.1, 0.05, 0.85
    } else {
        false
    }
}

/// Checks for Hammer: medium bullish body, long lower shadow, small upper shadow.
fn is_hammer(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, is_bullish)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        is_bullish && body_ratio > 0.05 && body_ratio <= 0.85 && // Relaxed from 0.1–0.8
        lower_shadow_ratio >= upper_shadow_ratio + 0.01 && // Relaxed from 0.02
        upper_shadow_ratio <= 0.15 // Relaxed from 0.1
    } else {
        false
    }
}

/// Checks for Hanging Man: medium bearish body, long lower shadow, small upper shadow.
fn is_hanging_man(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, is_bullish)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        !is_bullish && body_ratio > 0.05 && body_ratio <= 0.85 && // Relaxed from 0.1–0.8
        lower_shadow_ratio >= upper_shadow_ratio + 0.01 && // Relaxed from 0.02
        upper_shadow_ratio <= 0.15 // Relaxed from 0.1
    } else {
        false
    }
}

/// Checks for Inverted Hammer: medium bullish body, long upper shadow, small lower shadow.
fn is_inverted_hammer(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, is_bullish)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        is_bullish && body_ratio > 0.05 && body_ratio <= 0.85 && // Relaxed from 0.1–0.8
        upper_shadow_ratio >= lower_shadow_ratio + 0.01 && // Relaxed from 0.02
        lower_shadow_ratio <= 0.15 // Relaxed from 0.1
    } else {
        false
    }
}

/// Checks for Shooting Star: medium bearish body, long upper shadow, small lower shadow.
fn is_shooting_star(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, is_bullish)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        !is_bullish && body_ratio > 0.05 && body_ratio <= 0.85 && // Relaxed from 0.1–0.8
        upper_shadow_ratio >= lower_shadow_ratio + 0.01 && // Relaxed from 0.02
        lower_shadow_ratio <= 0.15 // Relaxed from 0.1
    } else {
        false
    }
}

/// Checks for Spinning Top: medium body, balanced shadows, neutral direction.
fn is_spinning_top(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        let shadow_diff = (upper_shadow_ratio - lower_shadow_ratio).abs();
        body_ratio > 0.05 && body_ratio <= 0.85 && // Relaxed from 0.1–0.8
        shadow_diff <= 0.05 && // Relaxed from 0.02
        upper_shadow_ratio >= 0.05 && lower_shadow_ratio >= 0.05 // Relaxed from 0.1
    } else {
        false
    }
}

/// Checks for Marubozu: minimal shadows, strong directional move.
fn is_marubozu(ohlc: &OHLC) -> bool {
    if let Some((total_range, _, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let upper_shadow_ratio = upper_shadow / total_range;
        let lower_shadow_ratio = lower_shadow / total_range;
        upper_shadow_ratio < 0.15 && lower_shadow_ratio < 0.15 // Relaxed from 0.1
    } else {
        false
    }
}

/// Analyzes single candlestick patterns for all OHLC entries.
fn analyze_single_candlestick_patterns(data: &mut Vec<OHLC>) {
    for ohlc in data.iter_mut() {
        if is_classic_doji(ohlc) {
            ohlc.patterns.insert("Classic_Doji".to_string());
        }
        if is_bullish_doji(ohlc) {
            ohlc.patterns.insert("Bullish_Doji".to_string());
        }
        if is_bearish_doji(ohlc) {
            ohlc.patterns.insert("Bearish_Doji".to_string());
        }
        if is_dragonfly_doji(ohlc) {
            ohlc.patterns.insert("Dragonfly_Doji".to_string());
        }
        if is_gravestone_doji(ohlc) {
            ohlc.patterns.insert("Gravestone_Doji".to_string());
        }
        if is_hammer(ohlc) {
            ohlc.patterns.insert("Hammer".to_string());
        }
        if is_hanging_man(ohlc) {
            ohlc.patterns.insert("Hanging_Man".to_string());
        }
        if is_inverted_hammer(ohlc) {
            ohlc.patterns.insert("Inverted_Hammer".to_string());
        }
        if is_shooting_star(ohlc) {
            ohlc.patterns.insert("Shooting_Star".to_string());
        }
        if is_spinning_top(ohlc) {
            ohlc.patterns.insert("Spinning_Top".to_string());
        }
        if is_marubozu(ohlc) {
            if ohlc.close > ohlc.open {
                ohlc.patterns.insert("Marubozu_Bullish".to_string());
            } else if ohlc.close < ohlc.open {
                ohlc.patterns.insert("Marubozu_Bearish".to_string());
            }
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

    // Analyze patterns
    analyze_single_candlestick_patterns(&mut data);

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
        "../single_candlestick_GBPAUD_daily.csv",
        "daily"
    );

    process_timeframe(
        "../GBPAUD_weekly.csv",
        "../single_candlestick_GBPAUD_weekly.csv",
        "weekly"
    );

    process_timeframe(
        "../GBPAUD_monthly.csv",
        "../single_candlestick_GBPAUD_monthly.csv",
        "monthly"
    );
    
    // Process NZDJPY datasets
    process_timeframe(
        "../NZDJPY_daily.csv",
        "../single_candlestick_NZDJPY_daily.csv",
        "daily"
    );

    process_timeframe(
        "../NZDJPY_weekly.csv",
        "../single_candlestick_NZDJPY_weekly.csv",
        "weekly"
    );

    process_timeframe(
        "../NZDJPY_monthly.csv",
        "../single_candlestick_NZDJPY_monthly.csv",
        "monthly"
    );
}