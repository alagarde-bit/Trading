use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
struct OHLC {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    daily_patterns: HashSet<String>,
    weekly_patterns: HashSet<String>,
    monthly_patterns: HashSet<String>,
    prev_pattern_daily: String,
    prev_3_patterns_daily: String,
    prev_10_patterns_daily: String,
    prev_pattern_weekly: String,
    prev_3_patterns_weekly: String,
    prev_10_patterns_weekly: String,
    prev_pattern_monthly: String,
    prev_3_patterns_monthly: String,
    prev_10_patterns_monthly: String,
}

fn parse_line(line: &str) -> Option<OHLC> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 8 {
        return None;
    }

    let daily_patterns = if parts[5].trim().is_empty() {
        HashSet::new()
    } else {
        parts[5].split(';').map(|s| s.trim().to_string()).collect()
    };

    let weekly_patterns = if parts[6].trim().is_empty() {
        HashSet::new()
    } else {
        parts[6].split(';').map(|s| s.trim().to_string()).collect()
    };

    let monthly_patterns = if parts[7].trim().is_empty() {
        HashSet::new()
    } else {
        parts[7].split(';').map(|s| s.trim().to_string()).collect()
    };

    Some(OHLC {
        date: parts[0].to_string(),
        open: parts[1].parse().ok()?,
        high: parts[2].parse().ok()?,
        low: parts[3].parse().ok()?,
        close: parts[4].parse().ok()?,
        daily_patterns,
        weekly_patterns,
        monthly_patterns,
        prev_pattern_daily: String::new(),
        prev_3_patterns_daily: String::new(),
        prev_10_patterns_daily: String::new(),
        prev_pattern_weekly: String::new(),
        prev_3_patterns_weekly: String::new(),
        prev_10_patterns_weekly: String::new(),
        prev_pattern_monthly: String::new(),
        prev_3_patterns_monthly: String::new(),
        prev_10_patterns_monthly: String::new(),
    })
}

fn add_historical_patterns(data: &mut Vec<OHLC>) {
    // For each timeframe, build a list of (date, patterns) for all rows with patterns
    let mut daily_history: Vec<(String, HashSet<String>)> = data.iter()
        .filter(|ohlc| !ohlc.daily_patterns.is_empty())
        .map(|ohlc| (ohlc.date.clone(), ohlc.daily_patterns.clone())).collect();
    let mut weekly_history: Vec<(String, HashSet<String>)> = data.iter()
        .filter(|ohlc| !ohlc.weekly_patterns.is_empty())
        .map(|ohlc| (ohlc.date.clone(), ohlc.weekly_patterns.clone())).collect();
    let mut monthly_history: Vec<(String, HashSet<String>)> = data.iter()
        .filter(|ohlc| !ohlc.monthly_patterns.is_empty())
        .map(|ohlc| (ohlc.date.clone(), ohlc.monthly_patterns.clone())).collect();
    
    daily_history.sort_by(|a, b| a.0.cmp(&b.0));
    weekly_history.sort_by(|a, b| a.0.cmp(&b.0));
    monthly_history.sort_by(|a, b| a.0.cmp(&b.0));

    for ohlc in data.iter_mut() {
        // Helper closure to get previous patterns for a given history
        let get_prev_patterns = |history: &Vec<(String, HashSet<String>)>, date: &String| -> (String, String, String) {
            // Find all patterns before the current date
            let prev_patterns: Vec<&HashSet<String>> = history.iter()
                .filter(|(hist_date, _)| hist_date < date)
                .map(|(_, patterns)| patterns)
                .collect();
            
            // Get the most recent pattern
            let prev_pattern = prev_patterns.last()
                .map(|patterns| {
                    let mut sorted: Vec<String> = patterns.iter().cloned().collect();
                    sorted.sort();
                    sorted.join(";")
                })
                .unwrap_or_default();
            
            // Get the last 3 patterns
            let prev_3_patterns = prev_patterns.iter()
                .rev()
                .take(3)
                .flat_map(|patterns| patterns.iter().cloned())
                .collect::<HashSet<String>>();
            let mut sorted_3: Vec<String> = prev_3_patterns.into_iter().collect();
            sorted_3.sort();
            let prev_3_patterns_str = sorted_3.join(";");
            
            // Get the last 10 patterns
            let prev_10_patterns = prev_patterns.iter()
                .rev()
                .take(10)
                .flat_map(|patterns| patterns.iter().cloned())
                .collect::<HashSet<String>>();
            let mut sorted_10: Vec<String> = prev_10_patterns.into_iter().collect();
            sorted_10.sort();
            let prev_10_patterns_str = sorted_10.join(";");
            
            (prev_pattern, prev_3_patterns_str, prev_10_patterns_str)
        };

        let (prev_pattern_daily, prev_3_patterns_daily, prev_10_patterns_daily) = get_prev_patterns(&daily_history, &ohlc.date);
        let (prev_pattern_weekly, prev_3_patterns_weekly, prev_10_patterns_weekly) = get_prev_patterns(&weekly_history, &ohlc.date);
        let (prev_pattern_monthly, prev_3_patterns_monthly, prev_10_patterns_monthly) = get_prev_patterns(&monthly_history, &ohlc.date);

        ohlc.prev_pattern_daily = prev_pattern_daily;
        ohlc.prev_3_patterns_daily = prev_3_patterns_daily;
        ohlc.prev_10_patterns_daily = prev_10_patterns_daily;
        ohlc.prev_pattern_weekly = prev_pattern_weekly;
        ohlc.prev_3_patterns_weekly = prev_3_patterns_weekly;
        ohlc.prev_10_patterns_weekly = prev_10_patterns_weekly;
        ohlc.prev_pattern_monthly = prev_pattern_monthly;
        ohlc.prev_3_patterns_monthly = prev_3_patterns_monthly;
        ohlc.prev_10_patterns_monthly = prev_10_patterns_monthly;
    }
}

fn main() {
    // Process GBPAUD
    println!("Processing GBPAUD...");
    process_historical_candlesticks(
        "../merged_candlesticks_GBPAUD.csv",
        "../historical_candlesticks_GBPAUD.csv"
    );
    
    // Process NZDJPY
    println!("Processing NZDJPY...");
    process_historical_candlesticks(
        "../merged_candlesticks_NZDJPY.csv",
        "../historical_candlesticks_NZDJPY.csv"
    );
}

fn process_historical_candlesticks(input_file: &str, output_file: &str) {
    println!("Reading merged candlestick patterns file: {}", input_file);
    let mut data = Vec::new();
    let file = File::open(input_file).expect(&format!("Failed to open {}", input_file));
    let reader = BufReader::new(file);
    let mut is_header = true;

    for line in reader.lines() {
        if let Ok(line) = line {
            if is_header {
                is_header = false;
                continue;
            }
            if let Some(ohlc) = parse_line(&line) {
                data.push(ohlc);
            }
        }
    }

    println!("Adding historical patterns...");
    add_historical_patterns(&mut data);

    println!("Writing output file: {}", output_file);
    let mut file = File::create(output_file).expect(&format!("Failed to create {}", output_file));
    writeln!(
        file,
        "date,open,high,low,close,daily_patterns,weekly_patterns,monthly_patterns,prev_pattern_daily,prev_3_patterns_daily,prev_10_patterns_daily,prev_pattern_weekly,prev_3_patterns_weekly,prev_10_patterns_weekly,prev_pattern_monthly,prev_3_patterns_monthly,prev_10_patterns_monthly"
    ).expect("Failed to write header");

    for ohlc in data {
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
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            ohlc.date,
            ohlc.open,
            ohlc.high,
            ohlc.low,
            ohlc.close,
            daily_patterns,
            weekly_patterns,
            monthly_patterns,
            ohlc.prev_pattern_daily,
            ohlc.prev_3_patterns_daily,
            ohlc.prev_10_patterns_daily,
            ohlc.prev_pattern_weekly,
            ohlc.prev_3_patterns_weekly,
            ohlc.prev_10_patterns_weekly,
            ohlc.prev_pattern_monthly,
            ohlc.prev_3_patterns_monthly,
            ohlc.prev_10_patterns_monthly
        ).expect("Failed to write line");
    }

    println!("Processing complete. Output written to {}", output_file);
} 