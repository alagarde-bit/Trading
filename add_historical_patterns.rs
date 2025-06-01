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
    patterns: HashSet<String>,
    prev_pattern: String,
    prev_3_patterns: String,
    prev_10_patterns: String,
}

fn parse_line(line: &str) -> Option<OHLC> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 6 {
        return None;
    }

    let patterns = if parts[5].trim().is_empty() {
        HashSet::new()
    } else {
        parts[5].split(';').map(|s| s.trim().to_string()).collect()
    };

    Some(OHLC {
        date: parts[0].to_string(),
        open: parts[1].parse().ok()?,
        high: parts[2].parse().ok()?,
        low: parts[3].parse().ok()?,
        close: parts[4].parse().ok()?,
        patterns,
        prev_pattern: String::new(),
        prev_3_patterns: String::new(),
        prev_10_patterns: String::new(),
    })
}

fn add_historical_patterns(data: &mut Vec<OHLC>) {
    // First build complete history of all pattern dates
    let mut all_pattern_dates: Vec<(String, HashSet<String>)> = data
        .iter()
        .filter(|ohlc| !ohlc.patterns.is_empty())
        .map(|ohlc| (ohlc.date.clone(), ohlc.patterns.clone()))
        .collect();
    
    // Sort by date to ensure chronological order
    all_pattern_dates.sort_by(|a, b| a.0.cmp(&b.0));
    
    // Process each OHLC entry
    for ohlc in data.iter_mut() {
        let mut prev_patterns = Vec::new();
        let mut prev_3_patterns = Vec::new();
        let mut prev_10_patterns = Vec::new();
        
        // Find all patterns before current date
        let mut pattern_count = 0;
        for (hist_date, hist_patterns) in all_pattern_dates.iter() {
            if hist_date >= &ohlc.date {
                break;
            }
            
            // Most recent pattern
            if prev_patterns.is_empty() {
                prev_patterns.extend(hist_patterns.iter().cloned());
            }
            
            // Last 3 patterns
            if pattern_count < 3 {
                prev_3_patterns.extend(hist_patterns.iter().cloned());
            }
            
            // Last 10 patterns
            if pattern_count < 10 {
                prev_10_patterns.extend(hist_patterns.iter().cloned());
            }
            
            pattern_count += 1;
        }
        
        // Sort patterns for consistent output
        prev_patterns.sort();
        prev_3_patterns.sort();
        prev_10_patterns.sort();
        
        // Remove duplicates while maintaining order
        prev_patterns.dedup();
        prev_3_patterns.dedup();
        prev_10_patterns.dedup();
        
        // Update OHLC with historical patterns
        ohlc.prev_pattern = prev_patterns.join(";");
        ohlc.prev_3_patterns = prev_3_patterns.join(";");
        ohlc.prev_10_patterns = prev_10_patterns.join(";");
    }
}

fn main() {
    println!("Reading merged patterns file...");
    let mut data = Vec::new();
    let file = File::open("../merged_patterns_GBPAUD_daily.csv").expect("Failed to open merged patterns file");
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

    println!("Writing output file...");
    let mut file = File::create("../historical_patterns_GBPAUD_daily.csv").expect("Failed to create output file");
    writeln!(
        file,
        "date,open,high,low,close,patterns,prev_pattern,prev_3_patterns,prev_10_patterns"
    ).expect("Failed to write header");

    for ohlc in data {
        let patterns = if ohlc.patterns.is_empty() {
            String::new()
        } else {
            let mut patterns: Vec<String> = ohlc.patterns.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{}",
            ohlc.date,
            ohlc.open,
            ohlc.high,
            ohlc.low,
            ohlc.close,
            patterns,
            ohlc.prev_pattern,
            ohlc.prev_3_patterns,
            ohlc.prev_10_patterns
        ).expect("Failed to write line");
    }

    println!("Processing complete. Output written to historical_patterns_GBPAUD_daily.csv");
} 