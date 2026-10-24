use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug)]
struct CombinedData {
    date: String,
    // Daily OHLC
    daily_open: f64,
    daily_high: f64,
    daily_low: f64,
    daily_close: f64,
    // Weekly OHLC
    weekly_open: f64,
    weekly_high: f64,
    weekly_low: f64,
    weekly_close: f64,
    // Monthly OHLC
    monthly_open: f64,
    monthly_high: f64,
    monthly_low: f64,
    monthly_close: f64,
    // Daily patterns
    daily_candlestick_patterns: String,
    daily_candlestick_prev_pattern: String,
    daily_candlestick_prev_3_patterns: String,
    daily_candlestick_prev_10_patterns: String,
    daily_chart_patterns: String,
    daily_chart_prev_pattern: String,
    daily_chart_prev_3_patterns: String,
    daily_chart_prev_10_patterns: String,
    // Weekly patterns
    weekly_candlestick_patterns: String,
    weekly_candlestick_prev_pattern: String,
    weekly_candlestick_prev_3_patterns: String,
    weekly_candlestick_prev_10_patterns: String,
    weekly_chart_patterns: String,
    weekly_chart_prev_pattern: String,
    weekly_chart_prev_3_patterns: String,
    weekly_chart_prev_10_patterns: String,
    // Monthly patterns
    monthly_candlestick_patterns: String,
    monthly_candlestick_prev_pattern: String,
    monthly_candlestick_prev_3_patterns: String,
    monthly_candlestick_prev_10_patterns: String,
    monthly_chart_patterns: String,
    monthly_chart_prev_pattern: String,
    monthly_chart_prev_3_patterns: String,
    monthly_chart_prev_10_patterns: String,
    // Daily indicators (20, 50, 200)
    daily_sma_20: String,
    daily_sma_50: String,
    daily_sma_200: String,
    daily_rsi: String,
    // Weekly indicators (4, 10, 40)
    weekly_sma_4: String,
    weekly_sma_10: String,
    weekly_sma_40: String,
    weekly_rsi: String,
    // Monthly indicators (1, 3, 10)
    monthly_sma_1: String,
    monthly_sma_3: String,
    monthly_sma_10: String,
    monthly_rsi: String,
}

fn parse_historical_candlesticks(line: &str) -> Option<(String, (String, String, String, String, String, String, String, String, String, String, String, String))> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 13 {
        return None;
    }

    Some((
        parts[0].to_string(),
        (
            parts[5].to_string(),  // daily patterns
            parts[6].to_string(),  // daily prev_pattern
            parts[7].to_string(),  // daily prev_3_patterns
            parts[8].to_string(),  // daily prev_10_patterns
            parts[9].to_string(),  // weekly patterns
            parts[10].to_string(), // weekly prev_pattern
            parts[11].to_string(), // weekly prev_3_patterns
            parts[12].to_string(), // weekly prev_10_patterns
            parts[13].to_string(), // monthly patterns
            parts[14].to_string(), // monthly prev_pattern
            parts[15].to_string(), // monthly prev_3_patterns
            parts[16].to_string(), // monthly prev_10_patterns
        ),
    ))
}

fn parse_historical_patterns(line: &str) -> Option<(String, (String, String, String, String, String, String, String, String, String, String, String, String))> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 13 {
        return None;
    }

    Some((
        parts[0].to_string(),
        (
            parts[5].to_string(),  // daily patterns
            parts[6].to_string(),  // daily prev_pattern
            parts[7].to_string(),  // daily prev_3_patterns
            parts[8].to_string(),  // daily prev_10_patterns
            parts[9].to_string(),  // weekly patterns
            parts[10].to_string(), // weekly prev_pattern
            parts[11].to_string(), // weekly prev_3_patterns
            parts[12].to_string(), // weekly prev_10_patterns
            parts[13].to_string(), // monthly patterns
            parts[14].to_string(), // monthly prev_pattern
            parts[15].to_string(), // monthly prev_3_patterns
            parts[16].to_string(), // monthly prev_10_patterns
        ),
    ))
}

fn parse_indicators(line: &str) -> Option<(String, (f64, f64, f64, f64, String, String, String, String))> {
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
            parts[1].parse().ok()?, // open
            parts[2].parse().ok()?, // high
            parts[3].parse().ok()?, // low
            parts[4].parse().ok()?, // close
            sma_parts[0].to_string(), // sma_1/4/20
            sma_parts[1].to_string(), // sma_3/10/50
            sma_parts[2].to_string(), // sma_10/40/200
            parts[6].to_string(), // rsi
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
        // Daily OHLC
        daily_open: parts[1].parse().ok()?,
        daily_high: parts[2].parse().ok()?,
        daily_low: parts[3].parse().ok()?,
        daily_close: parts[4].parse().ok()?,
        // Weekly OHLC
        weekly_open: 0.0,
        weekly_high: 0.0,
        weekly_low: 0.0,
        weekly_close: 0.0,
        // Monthly OHLC
        monthly_open: 0.0,
        monthly_high: 0.0,
        monthly_low: 0.0,
        monthly_close: 0.0,
        // Daily patterns
        daily_candlestick_patterns: String::new(),
        daily_candlestick_prev_pattern: String::new(),
        daily_candlestick_prev_3_patterns: String::new(),
        daily_candlestick_prev_10_patterns: String::new(),
        daily_chart_patterns: String::new(),
        daily_chart_prev_pattern: String::new(),
        daily_chart_prev_3_patterns: String::new(),
        daily_chart_prev_10_patterns: String::new(),
        // Weekly patterns
        weekly_candlestick_patterns: String::new(),
        weekly_candlestick_prev_pattern: String::new(),
        weekly_candlestick_prev_3_patterns: String::new(),
        weekly_candlestick_prev_10_patterns: String::new(),
        weekly_chart_patterns: String::new(),
        weekly_chart_prev_pattern: String::new(),
        weekly_chart_prev_3_patterns: String::new(),
        weekly_chart_prev_10_patterns: String::new(),
        // Monthly patterns
        monthly_candlestick_patterns: String::new(),
        monthly_candlestick_prev_pattern: String::new(),
        monthly_candlestick_prev_3_patterns: String::new(),
        monthly_candlestick_prev_10_patterns: String::new(),
        monthly_chart_patterns: String::new(),
        monthly_chart_prev_pattern: String::new(),
        monthly_chart_prev_3_patterns: String::new(),
        monthly_chart_prev_10_patterns: String::new(),
        // Daily indicators (20, 50, 200)
        daily_sma_20: String::new(),
        daily_sma_50: String::new(),
        daily_sma_200: String::new(),
        daily_rsi: String::new(),
        // Weekly indicators (4, 10, 40)
        weekly_sma_4: String::new(),
        weekly_sma_10: String::new(),
        weekly_sma_40: String::new(),
        weekly_rsi: String::new(),
        // Monthly indicators (1, 3, 10)
        monthly_sma_1: String::new(),
        monthly_sma_3: String::new(),
        monthly_sma_10: String::new(),
        monthly_rsi: String::new(),
    })
}

fn main() {
    // Process GBPAUD
    println!("Processing GBPAUD data combination...");
    process_combine_patterns_and_indicators("GBPAUD");
    
    // Process NZDJPY
    println!("Processing NZDJPY data combination...");
    process_combine_patterns_and_indicators("NZDJPY");

    // Process GBPUSD
    println!("Processing GBPUSD data combination...");
    process_combine_patterns_and_indicators("GBPUSD");

    // Process USDCHF
    println!("Processing USDCHF data combination...");
    process_combine_patterns_and_indicators("USDCHF");

    // Process USDJPY
    println!("Processing USDJPY data combination...");
    process_combine_patterns_and_indicators("USDJPY");
}

fn process_combine_patterns_and_indicators(currency: &str) {
    println!("Reading data files for {}...", currency);
    let mut combined_data = Vec::new();
    
    // Maps for patterns
    let mut candlestick_map: HashMap<String, (String, String, String, String, String, String, String, String, String, String, String, String)> = HashMap::new();
    let mut pattern_map: HashMap<String, (String, String, String, String, String, String, String, String, String, String, String, String)> = HashMap::new();
    let mut daily_indicators_map: HashMap<String, (f64, f64, f64, f64, String, String, String, String)> = HashMap::new();
    
    // Maps for weekly data
    let mut weekly_indicators_map: HashMap<String, (f64, f64, f64, f64, String, String, String, String)> = HashMap::new();
    let mut last_weekly_merge_data: Option<(f64, f64, f64, f64, String, String, String, String)> = None;
    
    // Maps for monthly data
    let mut monthly_indicators_map: HashMap<String, (f64, f64, f64, f64, String, String, String, String)> = HashMap::new();
    let mut last_monthly_merge_data: Option<(f64, f64, f64, f64, String, String, String, String)> = None;

    // Helper function to check if OHLC data is valid (not all zeros)
    fn is_valid_ohlc(data: &(f64, f64, f64, f64, String, String, String, String)) -> bool {
        data.0 != 0.0 || data.1 != 0.0 || data.2 != 0.0 || data.3 != 0.0
    }

    // Helper function to get the carried forward data for a given date
    fn get_carried_forward_data(
        date: &str,
        data_map: &HashMap<String, (f64, f64, f64, f64, String, String, String, String)>,
        last_merge_data: &mut Option<(f64, f64, f64, f64, String, String, String, String)>
    ) -> Option<(f64, f64, f64, f64, String, String, String, String)> {
        // First check if there's a new merge at this exact date
        if let Some(data) = data_map.get(date) {
            if is_valid_ohlc(data) {
                // Update the last merge data with this new data
                *last_merge_data = Some(data.clone());
                return Some(data.clone());
            }
        }
        
        // If no new merge at this date, return the last carried forward data
        if let Some(data) = last_merge_data {
            if is_valid_ohlc(data) {
                return Some(data.clone());
            }
        }
        
        None
    }

    // Read historical candlesticks
    println!("Reading historical candlesticks for {}...", currency);
    let candlestick_filename = format!("../historical_candlesticks_{}.csv", currency);
    let candlestick_file = File::open(&candlestick_filename)
        .expect(&format!("Failed to open historical candlesticks file {}", candlestick_filename));
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
    println!("Reading historical patterns for {}...", currency);
    let pattern_filename = format!("../historical_patterns_{}_all.csv", currency);
    let pattern_file = File::open(&pattern_filename)
        .expect(&format!("Failed to open historical patterns file {}", pattern_filename));
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

    // Read daily indicators
    println!("Reading daily indicators for {}...", currency);
    let indicators_filename = format!("../{}_daily_with_indicators.csv", currency);
    let indicators_file = File::open(&indicators_filename)
        .expect(&format!("Failed to open daily indicators file {}", indicators_filename));
    let reader = BufReader::new(indicators_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some((date, indicators)) = parse_indicators(&line) {
                daily_indicators_map.insert(date, indicators);
            }
        }
    }

    // Read weekly indicators
    println!("Reading weekly indicators for {}...", currency);
    let weekly_indicators_filename = format!("../{}_weekly_with_indicators.csv", currency);
    let weekly_indicators_file = File::open(&weekly_indicators_filename)
        .expect(&format!("Failed to open weekly indicators file {}", weekly_indicators_filename));
    let reader = BufReader::new(weekly_indicators_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some((date, indicators)) = parse_indicators(&line) {
                if is_valid_ohlc(&indicators) {
                    // Initialize the carried forward data with the first valid entry
                    if last_weekly_merge_data.is_none() {
                        last_weekly_merge_data = Some(indicators.clone());
                    }
                }
                weekly_indicators_map.insert(date, indicators);
            }
        }
    }

    // Read monthly indicators
    println!("Reading monthly indicators for {}...", currency);
    let monthly_indicators_filename = format!("../{}_monthly_with_indicators.csv", currency);
    let monthly_indicators_file = File::open(&monthly_indicators_filename)
        .expect(&format!("Failed to open monthly indicators file {}", monthly_indicators_filename));
    let reader = BufReader::new(monthly_indicators_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some((date, indicators)) = parse_indicators(&line) {
                if is_valid_ohlc(&indicators) {
                    // Initialize the carried forward data with the first valid entry
                    if last_monthly_merge_data.is_none() {
                        last_monthly_merge_data = Some(indicators.clone());
                    }
                }
                monthly_indicators_map.insert(date, indicators);
            }
        }
    }

    // Use daily indicators file as base for OHLC data since it has all dates
    println!("Combining all data for {}...", currency);
    let base_filename = format!("../{}_daily_with_indicators.csv", currency);
    let base_file = File::open(&base_filename)
        .expect(&format!("Failed to open base file {}", base_filename));
    let reader = BufReader::new(base_file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some(mut data) = parse_base_data(&line) {
                // Add daily candlestick patterns if available
                if let Some(candlesticks) = candlestick_map.get(&data.date) {
                    data.daily_candlestick_patterns = candlesticks.0.clone();
                    data.daily_candlestick_prev_pattern = candlesticks.1.clone();
                    data.daily_candlestick_prev_3_patterns = candlesticks.2.clone();
                    data.daily_candlestick_prev_10_patterns = candlesticks.3.clone();
                    data.weekly_candlestick_patterns = candlesticks.4.clone();
                    data.weekly_candlestick_prev_pattern = candlesticks.5.clone();
                    data.weekly_candlestick_prev_3_patterns = candlesticks.6.clone();
                    data.weekly_candlestick_prev_10_patterns = candlesticks.7.clone();
                    data.monthly_candlestick_patterns = candlesticks.8.clone();
                    data.monthly_candlestick_prev_pattern = candlesticks.9.clone();
                    data.monthly_candlestick_prev_3_patterns = candlesticks.10.clone();
                    data.monthly_candlestick_prev_10_patterns = candlesticks.11.clone();
                }

                // Add daily chart patterns if available
                if let Some(patterns) = pattern_map.get(&data.date) {
                    data.daily_chart_patterns = patterns.0.clone();
                    data.daily_chart_prev_pattern = patterns.1.clone();
                    data.daily_chart_prev_3_patterns = patterns.2.clone();
                    data.daily_chart_prev_10_patterns = patterns.3.clone();
                    data.weekly_chart_patterns = patterns.4.clone();
                    data.weekly_chart_prev_pattern = patterns.5.clone();
                    data.weekly_chart_prev_3_patterns = patterns.6.clone();
                    data.weekly_chart_prev_10_patterns = patterns.7.clone();
                    data.monthly_chart_patterns = patterns.8.clone();
                    data.monthly_chart_prev_pattern = patterns.9.clone();
                    data.monthly_chart_prev_3_patterns = patterns.10.clone();
                    data.monthly_chart_prev_10_patterns = patterns.11.clone();
                }

                // Add daily indicators if available
                if let Some(indicators) = daily_indicators_map.get(&data.date) {
                    data.daily_sma_20 = indicators.4.clone();
                    data.daily_sma_50 = indicators.5.clone();
                    data.daily_sma_200 = indicators.6.clone();
                    data.daily_rsi = indicators.7.clone();
                }

                // Add weekly indicators if available (using carried forward data)
                if let Some(indicators) = get_carried_forward_data(&data.date, &weekly_indicators_map, &mut last_weekly_merge_data) {
                    data.weekly_open = indicators.0;
                    data.weekly_high = indicators.1;
                    data.weekly_low = indicators.2;
                    data.weekly_close = indicators.3;
                    data.weekly_sma_4 = indicators.4.clone();
                    data.weekly_sma_10 = indicators.5.clone();
                    data.weekly_sma_40 = indicators.6.clone();
                    data.weekly_rsi = indicators.7.clone();
                }

                // Add monthly indicators if available (using carried forward data)
                if let Some(indicators) = get_carried_forward_data(&data.date, &monthly_indicators_map, &mut last_monthly_merge_data) {
                    data.monthly_open = indicators.0;
                    data.monthly_high = indicators.1;
                    data.monthly_low = indicators.2;
                    data.monthly_close = indicators.3;
                    data.monthly_sma_1 = indicators.4.clone();
                    data.monthly_sma_3 = indicators.5.clone();
                    data.monthly_sma_10 = indicators.6.clone();
                    data.monthly_rsi = indicators.7.clone();
                }

                combined_data.push(data);
            }
        }
    }

    println!("Writing combined output file for {}...", currency);
    let output_filename = format!("../combined_{}_daily.csv", currency);
    let mut file = File::create(&output_filename)
        .expect(&format!("Failed to create output file {}", output_filename));

    // Write header
    writeln!(
        file,
        "date,\
        daily_open,daily_high,daily_low,daily_close,\
        weekly_open,weekly_high,weekly_low,weekly_close,\
        monthly_open,monthly_high,monthly_low,monthly_close,\
        daily_candlestick_patterns,daily_candlestick_prev_pattern,daily_candlestick_prev_3_patterns,daily_candlestick_prev_10_patterns,\
        daily_chart_patterns,daily_chart_prev_pattern,daily_chart_prev_3_patterns,daily_chart_prev_10_patterns,\
        weekly_candlestick_patterns,weekly_candlestick_prev_pattern,weekly_candlestick_prev_3_patterns,weekly_candlestick_prev_10_patterns,\
        weekly_chart_patterns,weekly_chart_prev_pattern,weekly_chart_prev_3_patterns,weekly_chart_prev_10_patterns,\
        monthly_candlestick_patterns,monthly_candlestick_prev_pattern,monthly_candlestick_prev_3_patterns,monthly_candlestick_prev_10_patterns,\
        monthly_chart_patterns,monthly_chart_prev_pattern,monthly_chart_prev_3_patterns,monthly_chart_prev_10_patterns,\
        daily_sma_20,daily_sma_50,daily_sma_200,daily_rsi,\
        weekly_sma_4,weekly_sma_10,weekly_sma_40,weekly_rsi,\
        monthly_sma_1,monthly_sma_3,monthly_sma_10,monthly_rsi"
    ).expect("Failed to write header");

    // Write data
    for data in combined_data {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            data.date,
            data.daily_open, data.daily_high, data.daily_low, data.daily_close,
            data.weekly_open, data.weekly_high, data.weekly_low, data.weekly_close,
            data.monthly_open, data.monthly_high, data.monthly_low, data.monthly_close,
            data.daily_candlestick_patterns, data.daily_candlestick_prev_pattern,
            data.daily_candlestick_prev_3_patterns, data.daily_candlestick_prev_10_patterns,
            data.daily_chart_patterns, data.daily_chart_prev_pattern,
            data.daily_chart_prev_3_patterns, data.daily_chart_prev_10_patterns,
            data.weekly_candlestick_patterns, data.weekly_candlestick_prev_pattern,
            data.weekly_candlestick_prev_3_patterns, data.weekly_candlestick_prev_10_patterns,
            data.weekly_chart_patterns, data.weekly_chart_prev_pattern,
            data.weekly_chart_prev_3_patterns, data.weekly_chart_prev_10_patterns,
            data.monthly_candlestick_patterns, data.monthly_candlestick_prev_pattern,
            data.monthly_candlestick_prev_3_patterns, data.monthly_candlestick_prev_10_patterns,
            data.monthly_chart_patterns, data.monthly_chart_prev_pattern,
            data.monthly_chart_prev_3_patterns, data.monthly_chart_prev_10_patterns,
            data.daily_sma_20, data.daily_sma_50, data.daily_sma_200, data.daily_rsi,
            data.weekly_sma_4, data.weekly_sma_10, data.weekly_sma_40, data.weekly_rsi,
            data.monthly_sma_1, data.monthly_sma_3, data.monthly_sma_10, data.monthly_rsi
        ).expect("Failed to write line");
    }

    println!("Processing complete for {}. Output written to {}", currency, output_filename);
} 