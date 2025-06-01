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

/// Checks for Classic Doji (used for Morning/Evening Star Doji): small body, balanced shadows.
fn is_classic_doji(ohlc: &OHLC) -> bool {
    if let Some((total_range, body, upper_shadow, lower_shadow, _)) = compute_candlestick_metrics(ohlc) {
        let body_ratio = body / total_range;
        let shadow_diff = (upper_shadow - lower_shadow).abs() / total_range;
        body_ratio <= 0.165 && shadow_diff <= 0.055 // Increased by 10%: 0.15 → 0.165, 0.05 → 0.055
    } else {
        false
    }
}

/// Checks for Morning Star: bearish, small body, bullish.
fn is_morning_star(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar3_bullish = bar3.close > bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, _, _, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
                let body1_percent = body1 / bar1.open;
                let body3_percent = body3 / bar3.open;
                let body1_ratio = body1 / range1;
                let body2_ratio = body2 / range2;
                let body3_ratio = body3 / range3;
                let below_mid1 = bar2.close < (bar1.open + bar1.close) / 2.0;
                let below_mid3 = bar2.close < (bar3.open + bar3.close) / 2.0;
                is_bar1_bearish && is_bar3_bullish &&
                body1_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
                body1_ratio >= 0.51 && body3_ratio >= 0.51 &&
                body2_ratio <= 0.36 && body2 < body1 && // Relaxed by 10%: 0.40 → 0.36
                below_mid1 && below_mid3
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Evening Star: bullish, small body, bearish.
fn is_evening_star(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar3_bearish = bar3.close < bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, _, _, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
                let body1_percent = body1 / bar1.open;
                let body3_percent = body3 / bar3.open;
                let body1_ratio = body1 / range1;
                let body2_ratio = body2 / range2;
                let body3_ratio = body3 / range3;
                let above_mid1 = bar2.open > (bar1.open + bar1.close) / 2.0;
                let above_mid3 = bar2.open > (bar3.open + bar3.close) / 2.0;
                is_bar1_bullish && is_bar3_bearish &&
                body1_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
                body1_ratio >= 0.51 && body3_ratio >= 0.51 &&
                body2_ratio <= 0.36 && body2 < body1 && // Relaxed by 10%: 0.40 → 0.36
                above_mid1 && above_mid3
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Morning Star Doji: bearish, Doji, bullish.
fn is_morning_star_doji(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar3_bullish = bar3.close > bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
            let body1_percent = body1 / bar1.open;
            let body3_percent = body3 / bar3.open;
            let body1_ratio = body1 / range1;
            let body3_ratio = body3 / range3;
            let below_mid1 = bar2.close < (bar1.open + bar1.close) / 2.0;
            let below_mid3 = bar2.close < (bar3.open + bar3.close) / 2.0;
            is_bar1_bearish && is_bar3_bullish && is_classic_doji(bar2) &&
            body1_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
            body1_ratio >= 0.51 && body3_ratio >= 0.51 &&
            below_mid1 && below_mid3
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Evening Star Doji: bullish, Doji, bearish.
fn is_evening_star_doji(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar3_bearish = bar3.close < bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
            let body1_percent = body1 / bar1.open;
            let body3_percent = body3 / bar3.open;
            let body1_ratio = body1 / range1;
            let body3_ratio = body3 / range3;
            let above_mid1 = bar2.open > (bar1.open + bar1.close) / 2.0;
            let above_mid3 = bar2.open > (bar3.open + bar3.close) / 2.0;
            is_bar1_bullish && is_bar3_bearish && is_classic_doji(bar2) &&
            body1_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
            body1_ratio >= 0.51 && body3_ratio >= 0.51 &&
            above_mid1 && above_mid3
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Three White Soldiers: three consecutive bullish candles.
fn is_three_white_soldiers(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    let is_bar3_bullish = bar3.close > bar3.open;
    if let Some((range1, body1, upper_shadow1, lower_shadow1, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, upper_shadow2, lower_shadow2, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, upper_shadow3, lower_shadow3, _)) = compute_candlestick_metrics(bar3) {
                let body1_percent = body1 / bar1.open;
                let body2_percent = body2 / bar2.open;
                let body3_percent = body3 / bar3.open;
                let body1_ratio = body1 / range1;
                let body2_ratio = body2 / range2;
                let body3_ratio = body3 / range3;
                let upper_shadow1_ratio = upper_shadow1 / range1;
                let lower_shadow1_ratio = lower_shadow1 / range1;
                let upper_shadow2_ratio = upper_shadow2 / range2;
                let lower_shadow2_ratio = lower_shadow2 / range2;
                let upper_shadow3_ratio = upper_shadow3 / range3;
                let lower_shadow3_ratio = lower_shadow3 / range3;
                let close2_above_high1 = bar2.close > bar1.high;
                let close3_above_high2 = bar3.close > bar2.high;
                is_bar1_bullish && is_bar2_bullish && is_bar3_bullish &&
                body1_percent >= 0.0027 && body2_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
                body1_ratio >= 0.45 && body2_ratio >= 0.45 && body3_ratio >= 0.45 && // Relaxed by 10%: 0.50 → 0.45
                upper_shadow1_ratio <= 0.60 && lower_shadow1_ratio <= 0.60 && // Relaxed by 10%: 0.50 → 0.60
                upper_shadow2_ratio <= 0.60 && lower_shadow2_ratio <= 0.60 &&
                upper_shadow3_ratio <= 0.60 && lower_shadow3_ratio <= 0.60 &&
                close2_above_high1 && close3_above_high2
                // Removed: open2_within_body1 && open3_within_body2
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Three Black Crows: three consecutive bearish candles.
fn is_three_black_crows(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    let is_bar3_bearish = bar3.close < bar3.open;
    if let Some((range1, body1, upper_shadow1, lower_shadow1, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, upper_shadow2, lower_shadow2, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, upper_shadow3, lower_shadow3, _)) = compute_candlestick_metrics(bar3) {
                let body1_percent = body1 / bar1.open;
                let body2_percent = body2 / bar2.open;
                let body3_percent = body3 / bar3.open;
                let body1_ratio = body1 / range1;
                let body2_ratio = body2 / range2;
                let body3_ratio = body3 / range3;
                let upper_shadow1_ratio = upper_shadow1 / range1;
                let lower_shadow1_ratio = lower_shadow1 / range1;
                let upper_shadow2_ratio = upper_shadow2 / range2;
                let lower_shadow2_ratio = lower_shadow2 / range2;
                let upper_shadow3_ratio = upper_shadow3 / range3;
                let lower_shadow3_ratio = lower_shadow3 / range3;
                let close2_below_low1 = bar2.close < bar1.low;
                let close3_below_low2 = bar3.close < bar2.low;
                is_bar1_bearish && is_bar2_bearish && is_bar3_bearish &&
                body1_percent >= 0.0027 && body2_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
                body1_ratio >= 0.45 && body2_ratio >= 0.45 && body3_ratio >= 0.45 && // Relaxed by 10%: 0.50 → 0.45
                upper_shadow1_ratio <= 0.60 && lower_shadow1_ratio <= 0.60 && // Relaxed by 10%: 0.50 → 0.60
                upper_shadow2_ratio <= 0.60 && lower_shadow2_ratio <= 0.60 &&
                upper_shadow3_ratio <= 0.60 && lower_shadow3_ratio <= 0.60 &&
                close2_below_low1 && close3_below_low2
                // Removed: open2_within_body1 && open3_within_body2
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Three Outside Up: bearish, bullish engulfing, bullish confirmation.
fn is_three_outside_up(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    let is_bar3_bullish = bar3.close > bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, _, _, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
                let body3_percent = body3 / bar3.open;
                let body3_ratio = body3 / range3;
                let engulfing = body2 > body1 && bar2.open <= bar1.close && bar2.close >= bar1.open;
                let close3_above_close2 = bar3.close > bar2.close;
                is_bar1_bearish && is_bar2_bullish && is_bar3_bullish &&
                body3_percent >= 0.0027 && body3_ratio >= 0.45 && // Relaxed by 10%: 0.003 → 0.0027, 0.50 → 0.45
                engulfing && close3_above_close2
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Three Inside Up: bearish, bullish harami, bullish confirmation.
fn is_three_inside_up(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bearish = bar1.close < bar1.open;
    let is_bar2_bullish = bar2.close > bar2.open;
    let is_bar3_bullish = bar3.close > bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, _, _, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
                let body1_percent = body1 / bar1.open;
                let body3_percent = body3 / bar3.open;
                let body1_ratio = body1 / range1;
                let body3_ratio = body3 / range3;
                let harami = body2 < body1 && bar2.open >= bar1.close && bar2.close <= bar1.open;
                let close3_above_close2 = bar3.close > bar2.close;
                is_bar1_bearish && is_bar2_bullish && is_bar3_bullish &&
                body1_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
                body1_ratio >= 0.45 && body3_ratio >= 0.45 && // Relaxed by 10%: 0.50 → 0.45
                harami && close3_above_close2
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Three Outside Down: bullish, bearish engulfing, bearish confirmation.
fn is_three_outside_down(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    let is_bar3_bearish = bar3.close < bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, _, _, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
                let body3_percent = body3 / bar3.open;
                let body3_ratio = body3 / range3;
                let engulfing = body2 > body1 && bar2.open >= bar1.close && bar2.close <= bar1.open;
                let close3_below_close2 = bar3.close < bar2.close;
                is_bar1_bullish && is_bar2_bearish && is_bar3_bearish &&
                body3_percent >= 0.0027 && body3_ratio >= 0.45 && // Relaxed by 10%: 0.003 → 0.0027, 0.50 → 0.45
                engulfing && close3_below_close2
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Checks for Three Inside Down: bullish, bearish harami, bearish confirmation.
fn is_three_inside_down(data: &[OHLC], i: usize) -> bool {
    if i < 2 || i >= data.len() {
        return false;
    }
    let bar1 = &data[i - 2];
    let bar2 = &data[i - 1];
    let bar3 = &data[i];
    let is_bar1_bullish = bar1.close > bar1.open;
    let is_bar2_bearish = bar2.close < bar2.open;
    let is_bar3_bearish = bar3.close < bar3.open;
    if let Some((range1, body1, _, _, _)) = compute_candlestick_metrics(bar1) {
        if let Some((range2, body2, _, _, _)) = compute_candlestick_metrics(bar2) {
            if let Some((range3, body3, _, _, _)) = compute_candlestick_metrics(bar3) {
                let body1_percent = body1 / bar1.open;
                let body3_percent = body3 / bar3.open;
                let body1_ratio = body1 / range1;
                let body3_ratio = body3 / range3;
                let harami = body2 < body1 && bar2.open <= bar1.close && bar2.close >= bar1.open;
                let close3_below_close2 = bar3.close < bar2.close;
                is_bar1_bullish && is_bar2_bearish && is_bar3_bearish &&
                body1_percent >= 0.0027 && body3_percent >= 0.0027 && // Relaxed by 10%: 0.003 → 0.0027
                body1_ratio >= 0.45 && body3_ratio >= 0.45 && // Relaxed by 10%: 0.50 → 0.45
                harami && close3_below_close2
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

/// Analyzes triple candlestick patterns for all OHLC entries.
fn analyze_triple_candlestick_patterns(data: &mut Vec<OHLC>) {
    for i in 2..data.len() {
        if is_morning_star(data, i) {
            data[i - 2].patterns.insert("Morning_Star_1".to_string());
            data[i - 1].patterns.insert("Morning_Star_2".to_string());
            data[i].patterns.insert("Morning_Star_3".to_string());
        }
        if is_evening_star(data, i) {
            data[i - 2].patterns.insert("Evening_Star_1".to_string());
            data[i - 1].patterns.insert("Evening_Star_2".to_string());
            data[i].patterns.insert("Evening_Star_3".to_string());
        }
        if is_morning_star_doji(data, i) {
            data[i - 2].patterns.insert("Morning_Star_Doji_1".to_string());
            data[i - 1].patterns.insert("Morning_Star_Doji_2".to_string());
            data[i].patterns.insert("Morning_Star_Doji_3".to_string());
        }
        if is_evening_star_doji(data, i) {
            data[i - 2].patterns.insert("Evening_Star_Doji_1".to_string());
            data[i - 1].patterns.insert("Evening_Star_Doji_2".to_string());
            data[i].patterns.insert("Evening_Star_Doji_3".to_string());
        }
        if is_three_white_soldiers(data, i) {
            data[i - 2].patterns.insert("Three_White_Soldiers_1".to_string());
            data[i - 1].patterns.insert("Three_White_Soldiers_2".to_string());
            data[i].patterns.insert("Three_White_Soldiers_3".to_string());
        }
        if is_three_black_crows(data, i) {
            data[i - 2].patterns.insert("Three_Black_Crows_1".to_string());
            data[i - 1].patterns.insert("Three_Black_Crows_2".to_string());
            data[i].patterns.insert("Three_Black_Crows_3".to_string());
        }
        if is_three_outside_up(data, i) {
            data[i - 2].patterns.insert("Three_Outside_Up_1".to_string());
            data[i - 1].patterns.insert("Three_Outside_Up_2".to_string());
            data[i].patterns.insert("Three_Outside_Up_3".to_string());
        }
        if is_three_inside_up(data, i) {
            data[i - 2].patterns.insert("Three_Inside_Up_1".to_string());
            data[i - 1].patterns.insert("Three_Inside_Up_2".to_string());
            data[i].patterns.insert("Three_Inside_Up_3".to_string());
        }
        if is_three_outside_down(data, i) {
            data[i - 2].patterns.insert("Three_Outside_Down_1".to_string());
            data[i - 1].patterns.insert("Three_Outside_Down_2".to_string());
            data[i].patterns.insert("Three_Outside_Down_3".to_string());
        }
        if is_three_inside_down(data, i) {
            data[i - 2].patterns.insert("Three_Inside_Down_1".to_string());
            data[i - 1].patterns.insert("Three_Inside_Down_2".to_string());
            data[i].patterns.insert("Three_Inside_Down_3".to_string());
        }
    }
}

fn main() {
    // Read input CSV
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

    // Analyze triple candlestick patterns
    analyze_triple_candlestick_patterns(&mut data);

    // Write output CSV
    let mut file = File::create("../triple_candlestick_GBPAUD_daily.csv")
        .expect("Failed to create output file");
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