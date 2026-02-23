const fs = require('fs');
const readline = require('readline');

// Create readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Output file
const outputFile = 'indicator_data.json';

// Timeframes to collect data for
const timeframes = ['monthly', 'weekly', 'daily', '4h', '1h', '15m', '5m', '1m'];

// Indicators to collect
const indicators = ['volume', 'net volume', 'vwap'];

// Function to prompt user for input
function promptUser(question) {
  return new Promise((resolve) => rl.question(question, resolve));
}

// Function to collect data for a single timeframe
async function collectTimeframeData(timeframe) {
  const data = { timeframe, currency_pair: '', indicators: {} };
  
  console.log(`\nEntering data for ${timeframe}:`);
  
  // Prompt for currency pair
  const currencyPair = await promptUser('Currency Pair (e.g., EUR/USD, press Enter to skip): ');
  data.currency_pair = currencyPair.trim() || null;

  // Prompt for indicators
  for (const indicator of indicators) {
    const value = await promptUser(`${indicator} (enter a number, or press Enter to skip): `);
    data.indicators[indicator] = value.trim() === '' ? null : parseFloat(value) || 0;
  }
  
  return data;
}

// Function to append data to JSON file
function appendToJSON(newEntry) {
  let existingData = [];
  
  // Load existing data if file exists
  if (fs.existsSync(outputFile)) {
    const fileContent = fs.readFileSync(outputFile, 'utf8');
    existingData = JSON.parse(fileContent || '[]');
  }

  // Add timestamp and new entry
  newEntry.timestamp = new Date().toISOString();
  existingData.push(newEntry);

  // Write updated data back to file
  fs.writeFileSync(outputFile, JSON.stringify(existingData, null, 2), 'utf8');
  console.log(`Data saved to ${outputFile}. Total entries: ${existingData.length}`);
}

// Main execution
async function main() {
  try {
    console.log('Enter trading indicator data for each timeframe (press Ctrl+C to exit after saving).\n');
    
    const newData = [];
    for (const timeframe of timeframes) {
      const timeframeData = await collectTimeframeData(timeframe);
      newData.push(timeframeData);
    }

    // Append all timeframe data as a single batch
    for (const entry of newData) {
      appendToJSON(entry);
    }

    // Ask if user wants to enter another set
    const again = await promptUser('\nEnter another set of data? (y/n): ');
    if (again.toLowerCase() === 'y') {
      await main(); // Recursive call for another round
    } else {
      console.log('Exiting...');
      rl.close();
    }
  } catch (error) {
    console.error('Error:', error);
    rl.close();
  }
}

main();