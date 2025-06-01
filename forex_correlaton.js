const fs = require('fs');
const readline = require('readline');

// Create readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Output file
const outputFile = 'correlation_data.json';

// Currency pairs to track
const currencyPairs = ['NZD/JPY', 'GBP/AUD', 'USD/CHF', 'USD/JPY', 'GBP/USD'];

// Timeframes for correlations
const timeframes = ['monthly', 'weekly', 'daily'];

// Countries for 10-year bond yields (based on currencies: NZD, GBP, USD, JPY, AUD, CHF)
const bondYields = ['New Zealand', 'UK', 'US', 'Japan', 'Australia', 'Switzerland'];

// Function to prompt user for input
function promptUser(question) {
  return new Promise((resolve) => rl.question(question, resolve));
}

// Function to collect correlation data for a currency pair
async function collectCorrelationData(pair) {
  const correlations = {};
  console.log(`\nEntering correlations for ${pair}:`);
  
  for (const timeframe of timeframes) {
    const correlation = await promptUser(`${timeframe} correlation (-1 to 1, or press Enter to skip): `);
    correlations[timeframe] = correlation.trim() === '' ? null : parseFloat(correlation) || 0;
  }
  
  return correlations;
}

// Function to collect 10-year bond yield correlations
async function collectBondYieldCorrelations() {
  const correlations = {};
  console.log('\nEntering 10-year bond yield correlations with price:');
  
  for (const country of bondYields) {
    const correlation = await promptUser(`${country} 10-year bond yield correlation (-1 to 1, or press Enter to skip): `);
    correlations[country] = correlation.trim() === '' ? null : parseFloat(correlation) || 0;
  }
  
  return correlations;
}

// Function to collect all data
async function collectData() {
  const data = { currency_correlations: {}, bond_yield_correlations: {} };

  // Collect correlations for each currency pair
  for (const pair of currencyPairs) {
    data.currency_correlations[pair] = await collectCorrelationData(pair);
  }

  // Collect 10-year bond yield correlations
  data.bond_yield_correlations = await collectBondYieldCorrelations();

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
    console.log('Enter correlation data for currency pairs and 10-year bond yields (press Ctrl+C to exit after saving).\n');
    
    const newData = await collectData();
    appendToJSON(newData);

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