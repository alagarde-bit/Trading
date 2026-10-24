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

fn is_concluding_pattern(pattern: &str) -> bool {
    // Basic Patterns
    pattern.contains("Rectangle_Max4") ||
    pattern.contains("Rectangle_Min4") ||
    pattern.contains("DoubleTop_Max3") ||
    pattern.contains("DoubleBottom_Min3") ||
    pattern.contains("TripleTop_Max5") ||
    pattern.contains("TripleBottom_Min5") ||
    
    // Triangle and Broadening Patterns
    pattern.contains("Symmetrical_M3") ||
    pattern.contains("Ascending_M3") ||
    pattern.contains("Descending_X3") ||
    pattern.contains("Broadening_M3") ||
    pattern.contains("TowerTop_Bear2") ||
    pattern.contains("TowerBottom_Bull2") ||
    
    // Cup and Wedge Patterns
    pattern.contains("Cup_X2") ||
    pattern.contains("CupAndHandle_X3") ||
    pattern.contains("RisingWedge_M3") ||
    pattern.contains("FallingWedge_X3") ||
    pattern.contains("DeadCatBounce_M2") ||
    
    // Flag and Trend Patterns
    pattern.contains("Staircase_M2") ||
    pattern.contains("BullishFlag_X3") ||
    pattern.contains("BearishFlag_M3") ||
    pattern.contains("RisingTops_X3") ||
    pattern.contains("RisingBottoms_M3") ||
    pattern.contains("FallingTops_X3") ||
    pattern.contains("FallingBottoms_M3") ||
    
    // Head and Shoulders Patterns
    pattern.contains("HeadAndShoulders_RightShoulder") ||
    pattern.contains("InverseHeadAndShoulders_RightShoulder")
}

fn get_pattern_base(pattern: &str) -> String {
    // Extract pattern base name without the component identifier
    if let Some(idx) = pattern.find("_P") {
        let base = &pattern[..idx];
        // Get the pattern type without the component number
        if let Some(last_underscore) = base.rfind('_') {
            let pattern_type = &base[..last_underscore];
            return pattern_type.to_string();
        }
    }
    pattern.to_string()
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
            .filter(|s| is_concluding_pattern(s))
            .collect()
    };

    Some(OHLC {
        date: parts[0].to_string(),
        open: parts[1].parse().ok()?,
        high: parts[2].parse().ok()?,
        low: parts[3].parse().ok()?,
        close: parts[4].parse().ok()?,
        daily_patterns: if timeframe == "daily" { patterns.clone() } else { HashSet::new() },
        weekly_patterns: if timeframe == "weekly" { patterns.clone() } else { HashSet::new() },
        monthly_patterns: if timeframe == "monthly" { patterns } else { HashSet::new() },
    })
}

fn read_file(filename: &str) -> HashMap<String, OHLC> {
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
    
    // Determine timeframe from filename
    let timeframe = if filename.contains("_daily.csv") {
        "daily"
    } else if filename.contains("_weekly.csv") {
        "weekly"
    } else if filename.contains("_monthly.csv") {
        "monthly"
    } else {
        "daily" // default to daily if not specified
    };

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

fn merge_patterns(files: &[&str]) -> HashMap<String, OHLC> {
    let mut merged_data = HashMap::new();

    // Read first file as base
    if let Some(&first_file) = files.first() {
        merged_data = read_file(first_file);
    }

    // Merge patterns from other files
    for &file in files.iter().skip(1) {
        let current_data = read_file(file);
        
        for (date, ohlc) in current_data {
            if let Some(existing) = merged_data.get_mut(&date) {
                // Merge patterns based on timeframe
                if file.contains("_daily.csv") {
                    existing.daily_patterns.extend(ohlc.daily_patterns);
                } else if file.contains("_weekly.csv") {
                    existing.weekly_patterns.extend(ohlc.weekly_patterns);
                } else if file.contains("_monthly.csv") {
                    existing.monthly_patterns.extend(ohlc.monthly_patterns);
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
    println!("Processing GBPAUD chart patterns...");
    process_merge_patterns("GBPAUD");
    
    // Process NZDJPY
    println!("Processing NZDJPY chart patterns...");
    process_merge_patterns("NZDJPY");

    // Process GBPUSD
    println!("Processing GBPUSD chart patterns...");
    process_merge_patterns("GBPUSD");

    // Process USDCHF
    println!("Processing USDCHF chart patterns...");
    process_merge_patterns("USDCHF");

    // Process USDJPY
    println!("Processing USDJPY chart patterns...");
    process_merge_patterns("USDJPY");
}

fn process_merge_patterns(currency: &str) {
    let input_files = [
        // Daily files
        format!("../intermediate_{}_daily.csv", currency),
        format!("../intermediate2_{}_daily.csv", currency),
        format!("../new_patterns_{}_daily.csv", currency),
        format!("../chart_patterns_{}_daily.csv", currency),
        format!("../head_and_shoulders_{}_daily.csv", currency),
        // Weekly files
        format!("../intermediate_{}_weekly.csv", currency),
        format!("../intermediate2_{}_weekly.csv", currency),
        format!("../new_patterns_{}_weekly.csv", currency),
        format!("../chart_patterns_{}_weekly.csv", currency),
        format!("../head_and_shoulders_{}_weekly.csv", currency),
        // Monthly files
        format!("../intermediate_{}_monthly.csv", currency),
        format!("../intermediate2_{}_monthly.csv", currency),
        format!("../new_patterns_{}_monthly.csv", currency),
        format!("../chart_patterns_{}_monthly.csv", currency),
        format!("../head_and_shoulders_{}_monthly.csv", currency)
    ];

    let input_file_strs: Vec<&str> = input_files.iter().map(|s| s.as_str()).collect();

    println!("Starting pattern merge for {}...", currency);
    let merged_data = merge_patterns(&input_file_strs);
    println!("Merged {} records for {}", merged_data.len(), currency);

    // Write merged data to output file
    let output_filename = format!("../merged_patterns_{}_all.csv", currency);
    let mut output_file = File::create(&output_filename)
        .expect(&format!("Failed to create output file {}", output_filename));

    // Write header
    writeln!(output_file, "date,open,high,low,close,daily_patterns,weekly_patterns,monthly_patterns")
        .expect("Failed to write header");

    // Write data sorted by date
    let mut sorted_data: Vec<_> = merged_data.into_iter().collect();
    sorted_data.sort_by(|(date1, _), (date2, _)| date1.cmp(date2));

    for (_, ohlc) in sorted_data {
        let daily_pattern_str = if ohlc.daily_patterns.is_empty() {
            String::new()
        } else {
            let mut patterns: Vec<String> = ohlc.daily_patterns.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };

        let weekly_pattern_str = if ohlc.weekly_patterns.is_empty() {
            String::new()
        } else {
            let mut patterns: Vec<String> = ohlc.weekly_patterns.into_iter().collect();
            patterns.sort();
            patterns.join(";")
        };

        let monthly_pattern_str = if ohlc.monthly_patterns.is_empty() {
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
            daily_pattern_str, weekly_pattern_str, monthly_pattern_str
        ).expect("Failed to write data line");
    }

    println!("Merge complete for {}. Output written to {}", currency, output_filename);
} 