use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug)]
struct OHLC {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    daily_patterns: HashSet<String>,
    weekly_patterns: HashSet<String>,
    monthly_patterns: HashSet<String>,
}

fn is_confirming_candlestick(pattern: &str) -> bool {
    // Single candlestick patterns (complete on their own)
    pattern == "Classic_Doji" ||
    pattern == "Bullish_Doji" ||
    pattern == "Bearish_Doji" ||
    pattern == "Dragonfly_Doji" ||
    pattern == "Gravestone_Doji" ||
    pattern == "Hammer" ||
    pattern == "Hanging_Man" ||
    pattern == "Inverted_Hammer" ||
    pattern == "Shooting_Star" ||
    pattern == "Spinning_Top" ||
    pattern == "Marubozu_Bullish" ||
    pattern == "Marubozu_Bearish" ||

    // Double candlestick confirming patterns (second component)
    pattern == "Bullish_Kicker_2" ||
    pattern == "Bearish_Kicker_2" ||
    pattern == "Tweezer_Bottoms_2" ||
    pattern == "Tweezer_Tops_2" ||
    pattern == "Dark_Cloud_Cover_2" ||
    pattern == "Piercing_Line_2" ||
    pattern == "Double_Doji_2" ||
    pattern == "Matching_Low_2" ||
    pattern == "Matching_High_2" ||
    pattern == "Rising_Window_2" ||
    pattern == "Falling_Window_2" ||

    // Triple candlestick confirming patterns (third component)
    pattern == "Morning_Star_3" ||
    pattern == "Evening_Star_3" ||
    pattern == "Morning_Star_Doji_3" ||
    pattern == "Evening_Star_Doji_3" ||
    pattern == "Three_White_Soldiers_3" ||
    pattern == "Three_Black_Crows_3" ||
    pattern == "Three_Outside_Up_3" ||
    pattern == "Three_Inside_Up_3" ||
    pattern == "Three_Outside_Down_3" ||
    pattern == "Three_Inside_Down_3"
}

fn parse_line(line: &str, timeframe: &str) -> Option<OHLC> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 6 {
        return None;
    }

    let patterns = if parts[5].trim().is_empty() {
        HashSet::new()
    } else {
        parts[5]
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| is_confirming_candlestick(s))
            .collect()
    };

    let mut ohlc = OHLC {
        date: parts[0].to_string(),
        open: parts[1].parse().ok()?,
        high: parts[2].parse().ok()?,
        low: parts[3].parse().ok()?,
        close: parts[4].parse().ok()?,
        daily_patterns: HashSet::new(),
        weekly_patterns: HashSet::new(),
        monthly_patterns: HashSet::new(),
    };

    match timeframe {
        "daily" => ohlc.daily_patterns = patterns,
        "weekly" => ohlc.weekly_patterns = patterns,
        "monthly" => ohlc.monthly_patterns = patterns,
        _ => {}
    }

    Some(ohlc)
}

fn read_file(filename: &str, timeframe: &str) -> HashMap<String, OHLC> {
    let mut data = HashMap::new();
    let file = match File::open(filename) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Warning: Could not open file {}: {}", filename, e);
            return data;
        }
    };

    let reader = BufReader::new(file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some(ohlc) = parse_line(&line, timeframe) {
                data.insert(ohlc.date.clone(), ohlc);
            }
        }
    }

    data
}

fn merge_patterns(files: &[(&str, &str)]) -> HashMap<String, OHLC> {
    let mut merged_data = HashMap::new();

    // Read first file as base
    if let Some(&(first_file, timeframe)) = files.first() {
        merged_data = read_file(first_file, timeframe);
    }

    // Merge patterns from other files
    for &(file, timeframe) in files.iter().skip(1) {
        let current_data = read_file(file, timeframe);
        
        for (date, ohlc) in current_data {
            if let Some(existing) = merged_data.get_mut(&date) {
                // Merge patterns based on timeframe
                match timeframe {
                    "daily" => existing.daily_patterns.extend(ohlc.daily_patterns),
                    "weekly" => existing.weekly_patterns.extend(ohlc.weekly_patterns),
                    "monthly" => existing.monthly_patterns.extend(ohlc.monthly_patterns),
                    _ => {}
                }
            } else {
                // Add new entry
                merged_data.insert(date, ohlc);
            }
        }
    }

    merged_data
}

fn main() {
    // Process GBPAUD
    println!("Processing GBPAUD candlestick patterns...");
    process_merge_candlesticks("GBPAUD");
    
    // Process NZDJPY
    println!("Processing NZDJPY candlestick patterns...");
    process_merge_candlesticks("NZDJPY");
}

fn process_merge_candlesticks(currency: &str) {
    let input_files = [
        (format!("../single_candlestick_{}_daily.csv", currency), "daily"),
        (format!("../double_candlestick_{}_daily.csv", currency), "daily"),
        (format!("../triple_candlestick_{}_daily.csv", currency), "daily"),
        (format!("../single_candlestick_{}_weekly.csv", currency), "weekly"),
        (format!("../double_candlestick_{}_weekly.csv", currency), "weekly"),
        (format!("../triple_candlestick_{}_weekly.csv", currency), "weekly"),
        (format!("../single_candlestick_{}_monthly.csv", currency), "monthly"),
        (format!("../double_candlestick_{}_monthly.csv", currency), "monthly"),
        (format!("../triple_candlestick_{}_monthly.csv", currency), "monthly"),
    ];

    let input_file_refs: Vec<(&str, &str)> = input_files.iter().map(|(s, tf)| (s.as_str(), *tf)).collect();

    println!("Starting candlestick pattern merge for {}...", currency);
    let merged_data = merge_patterns(&input_file_refs);
    println!("Merged {} records for {}", merged_data.len(), currency);

    // Write merged data to output file
    let output_filename = format!("../merged_candlesticks_{}.csv", currency);
    let mut output_file = File::create(&output_filename)
        .expect(&format!("Failed to create output file {}", output_filename));

    // Write header
    writeln!(output_file, "date,open,high,low,close,daily_patterns,weekly_patterns,monthly_patterns")
        .expect("Failed to write header");

    // Write data sorted by date
    let mut sorted_data: Vec<_> = merged_data.into_iter().collect();
    sorted_data.sort_by(|(date1, _), (date2, _)| date1.cmp(date2));

    for (_, ohlc) in sorted_data {
        let daily_patterns = if ohlc.daily_patterns.is_empty() {
            String::new()
        } else {
            let mut patterns: Vec<String> = ohlc.daily_patterns.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };

        let weekly_patterns = if ohlc.weekly_patterns.is_empty() {
            String::new()
        } else {
            let mut patterns: Vec<String> = ohlc.weekly_patterns.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };

        let monthly_patterns = if ohlc.monthly_patterns.is_empty() {
            String::new()
        } else {
            let mut patterns: Vec<String> = ohlc.monthly_patterns.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };

        writeln!(
            output_file,
            "{},{},{},{},{},{},{},{}",
            ohlc.date, ohlc.open, ohlc.high, ohlc.low, ohlc.close,
            daily_patterns, weekly_patterns, monthly_patterns
        ).expect("Failed to write data line");
    }

    println!("Merge complete for {}. Output written to {}", currency, output_filename);
} 