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

fn process_dataset(input_file: &str, output_file: &str, timeframe: &str) {
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

    // Set SMA periods based on timeframe
    let (sma1, sma2, sma3) = match timeframe {
        "weekly" => (4, 10, 40),    // 4, 10, 40 weeks
        "monthly" => (1, 3, 10),    // 1, 3, 10 months
        _ => (20, 50, 200),         // Default daily periods
    };

    // Calculate SMAs and RSI
    let mut output_data: Vec<(String, f64, f64, f64, f64, String, f64)> = Vec::new();
    for i in 0..data.len() {
        let sma_1 = calculate_sma(&data, i, sma1);
        let sma_2 = calculate_sma(&data, i, sma2);
        let sma_3 = calculate_sma(&data, i, sma3);
        let sma_string = format!(
            "{};{};{}",
            if sma_1.is_nan() { "NaN".to_string() } else { format!("{:.4}", sma_1) },
            if sma_2.is_nan() { "NaN".to_string() } else { format!("{:.4}", sma_2) },
            if sma_3.is_nan() { "NaN".to_string() } else { format!("{:.4}", sma_3) }
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
    let mut file = File::create(output_file)
        .expect(&format!("Failed to create {}", output_file));
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

fn main() {
    // Process GBPAUD datasets
    process_dataset("../GBPAUD_daily.csv", "../GBPAUD_daily_with_indicators.csv", "daily");
    process_dataset("../GBPAUD_weekly.csv", "../GBPAUD_weekly_with_indicators.csv", "weekly");
    process_dataset("../GBPAUD_monthly.csv", "../GBPAUD_monthly_with_indicators.csv", "monthly");
    
    // Process NZDJPY datasets
    process_dataset("../NZDJPY_daily.csv", "../NZDJPY_daily_with_indicators.csv", "daily");
    process_dataset("../NZDJPY_weekly.csv", "../NZDJPY_weekly_with_indicators.csv", "weekly");
    process_dataset("../NZDJPY_monthly.csv", "../NZDJPY_monthly_with_indicators.csv", "monthly");
}