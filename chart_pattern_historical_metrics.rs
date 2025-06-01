use std::fs::File;
use std::io::{BufReader, Read, Write};

/// Represents a candlestick with OHLC data.
#[derive(Debug)]
struct OHLC {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
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
    })
}

/// Calculates Simple Moving Average (SMA) for a given period.
fn calculate_sma(data: &[OHLC], index: usize, period: usize) -> f64 {
    if index < period - 1 || index >= data.len() {
        return f64::NAN;
    }
    let slice = &data[index + 1 - period..=index];
    let sum: f64 = slice.iter().map(|ohlc| ohlc.close).sum();
    sum / period as f64
}

/// Calculates 14-period Relative Strength Index (RSI).
fn calculate_rsi(data: &[OHLC], index: usize, period: usize) -> f64 {
    if index < period - 1 || index >= data.len() {
        return f64::NAN;
    }
    let mut gains = 0.0;
    let mut losses = 0.0;
    let slice = &data[index + 1 - period..=index];

    for i in 1..slice.len() {
        let change = slice[i].close - slice[i - 1].close;
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        if avg_gain == 0.0 {
            return 50.0; // No movement, neutral RSI
        }
        return 100.0; // Only gains, RSI = 100
    }

    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
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

    // Calculate SMAs and RSI
    let mut output_data: Vec<(String, f64, f64, f64, f64, String, f64)> = Vec::new();
    for i in 0..data.len() {
        let sma_20 = calculate_sma(&data, i, 20);
        let sma_50 = calculate_sma(&data, i, 50);
        let sma_200 = calculate_sma(&data, i, 200);
        let sma_string = format!(
            "{};{};{}",
            if sma_20.is_nan() { "NaN".to_string() } else { format!("{:.4}", sma_20) },
            if sma_50.is_nan() { "NaN".to_string() } else { format!("{:.4}", sma_50) },
            if sma_200.is_nan() { "NaN".to_string() } else { format!("{:.4}", sma_200) }
        );
        let rsi_14 = calculate_rsi(&data, i, 14);
        let rsi_value = if rsi_14.is_nan() { f64::NAN } else { rsi_14 };
        output_data.push((
            data[i].date.clone(),
            data[i].open,
            data[i].high,
            data[i].low,
            data[i].close,
            sma_string,
            rsi_value,
        ));
    }

    // Write output CSV
    let mut file = File::create("../GBPAUD_daily_with_indicators.csv")
        .expect("Failed to create output file");
    writeln!(file, "date,open,high,low,close,sma,rsi").expect("Failed to write header");

    for entry in output_data {
        let rsi_str = if entry.6.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.2}", entry.6)
        };
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            entry.0, entry.1, entry.2, entry.3, entry.4, entry.5, rsi_str
        )
        .expect("Failed to write line");
    }
}