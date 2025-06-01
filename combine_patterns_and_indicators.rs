use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug)]
struct CombinedData {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    // Candlestick patterns
    candlestick_patterns: String,
    candlestick_prev_pattern: String,
    candlestick_prev_3_patterns: String,
    candlestick_prev_10_patterns: String,
    // Chart patterns
    chart_patterns: String,
    chart_prev_pattern: String,
    chart_prev_3_patterns: String,
    chart_prev_10_patterns: String,
    // Technical indicators
    sma_20: String,
    sma_50: String,
    sma_200: String,
    rsi: String,
}

fn parse_historical_candlesticks(line: &str) -> Option<(String, (String, String, String, String))> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 9 {
        return None;
    }

    Some((
        parts[0].to_string(),
        (
            parts[5].to_string(), // patterns
            parts[6].to_string(), // prev_pattern
            parts[7].to_string(), // prev_3_patterns
            parts[8].to_string(), // prev_10_patterns
        ),
    ))
}

fn parse_historical_patterns(line: &str) -> Option<(String, (String, String, String, String))> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 9 {
        return None;
    }

    Some((
        parts[0].to_string(),
        (
            parts[5].to_string(), // patterns
            parts[6].to_string(), // prev_pattern
            parts[7].to_string(), // prev_3_patterns
            parts[8].to_string(), // prev_10_patterns
        ),
    ))
}

fn parse_indicators(line: &str) -> Option<(String, (String, String, String, String))> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 7 {
        return None;
    }

    let sma_parts: Vec<&str> = parts[5].split(';').collect();
    if sma_parts.len() != 3 {
        return None;
    }

    Some((
        parts[0].to_string(),
        (
            sma_parts[0].to_string(),
            sma_parts[1].to_string(),
            sma_parts[2].to_string(),
            parts[6].to_string(),
        ),
    ))
}

fn parse_base_data(line: &str) -> Option<CombinedData> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 5 {
        return None;
    }

    Some(CombinedData {
        date: parts[0].to_string(),
        open: parts[1].parse().ok()?,
        high: parts[2].parse().ok()?,
        low: parts[3].parse().ok()?,
        close: parts[4].parse().ok()?,
        candlestick_patterns: String::new(),
        candlestick_prev_pattern: String::new(),
        candlestick_prev_3_patterns: String::new(),
        candlestick_prev_10_patterns: String::new(),
        chart_patterns: String::new(),
        chart_prev_pattern: String::new(),
        chart_prev_3_patterns: String::new(),
        chart_prev_10_patterns: String::new(),
        sma_20: String::new(),
        sma_50: String::new(),
        sma_200: String::new(),
        rsi: String::new(),
    })
}

fn main() {
    println!("Reading data files...");
    let mut combined_data = Vec::new();
    let mut candlestick_map = HashMap::new();
    let mut pattern_map = HashMap::new();
    let mut indicators_map = HashMap::new();

    // Read historical candlesticks
    println!("Reading historical candlesticks...");
    let candlestick_file = File::open("../historical_candlesticks_GBPAUD_daily.csv")
        .expect("Failed to open historical candlesticks file");
    let reader = BufReader::new(candlestick_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some((date, patterns)) = parse_historical_candlesticks(&line) {
                candlestick_map.insert(date, patterns);
            }
        }
    }

    // Read historical patterns
    println!("Reading historical patterns...");
    let pattern_file = File::open("../historical_patterns_GBPAUD_daily.csv")
        .expect("Failed to open historical patterns file");
    let reader = BufReader::new(pattern_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some((date, patterns)) = parse_historical_patterns(&line) {
                pattern_map.insert(date, patterns);
            }
        }
    }

    // Read indicators
    println!("Reading indicators...");
    let indicators_file = File::open("../GBPAUD_daily_with_indicators.csv")
        .expect("Failed to open indicators file");
    let reader = BufReader::new(indicators_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some((date, indicators)) = parse_indicators(&line) {
                indicators_map.insert(date, indicators);
            }
        }
    }

    // Use indicators file as base for OHLC data since it has all dates
    println!("Combining all data...");
    let base_file = File::open("../GBPAUD_daily_with_indicators.csv")
        .expect("Failed to open base file");
    let reader = BufReader::new(base_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some(mut data) = parse_base_data(&line) {
                // Add candlestick patterns if available
                if let Some(candlesticks) = candlestick_map.get(&data.date) {
                    data.candlestick_patterns = candlesticks.0.clone();
                    data.candlestick_prev_pattern = candlesticks.1.clone();
                    data.candlestick_prev_3_patterns = candlesticks.2.clone();
                    data.candlestick_prev_10_patterns = candlesticks.3.clone();
                }

                // Add chart patterns if available
                if let Some(patterns) = pattern_map.get(&data.date) {
                    data.chart_patterns = patterns.0.clone();
                    data.chart_prev_pattern = patterns.1.clone();
                    data.chart_prev_3_patterns = patterns.2.clone();
                    data.chart_prev_10_patterns = patterns.3.clone();
                }

                // Add indicators if available
                if let Some(indicators) = indicators_map.get(&data.date) {
                    data.sma_20 = indicators.0.clone();
                    data.sma_50 = indicators.1.clone();
                    data.sma_200 = indicators.2.clone();
                    data.rsi = indicators.3.clone();
                }

                combined_data.push(data);
            }
        }
    }

    println!("Writing combined output file...");
    let mut file = File::create("../combined_GBPAUD_daily.csv")
        .expect("Failed to create output file");

    // Write header
    writeln!(
        file,
        "date,open,high,low,close,candlestick_patterns,candlestick_prev_pattern,candlestick_prev_3_patterns,candlestick_prev_10_patterns,chart_patterns,chart_prev_pattern,chart_prev_3_patterns,chart_prev_10_patterns,sma_20,sma_50,sma_200,rsi"
    ).expect("Failed to write header");

    // Write data
    for data in combined_data {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            data.date,
            data.open,
            data.high,
            data.low,
            data.close,
            data.candlestick_patterns,
            data.candlestick_prev_pattern,
            data.candlestick_prev_3_patterns,
            data.candlestick_prev_10_patterns,
            data.chart_patterns,
            data.chart_prev_pattern,
            data.chart_prev_3_patterns,
            data.chart_prev_10_patterns,
            data.sma_20,
            data.sma_50,
            data.sma_200,
            data.rsi
        ).expect("Failed to write line");
    }

    println!("Processing complete. Output written to combined_GBPAUD_daily.csv");
} 