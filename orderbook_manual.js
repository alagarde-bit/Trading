const fs = require('fs');
const readline = require('readline');

// Create readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Output file
const outputFile = 'order_position_data.json';

// Function to prompt user for input
function promptUser(question) {
  return new Promise((resolve) => rl.question(question, resolve));
}

// Function to collect a price range (start and end)
async function collectPriceRange(rangeDescription) {
  const start = await promptUser(`Start price for ${rangeDescription} (or press Enter to skip): `);
  const end = await promptUser(`End price for ${rangeDescription} (or press Enter to skip): `);
  if (start.trim() === '' || end.trim() === '') return null;
  return { start: parseFloat(start) || 0, end: parseFloat(end) || 0 };
}

// Function to collect notable order data (dictionary)
async function collectNotableOrders() {
  const notableOrders = {};
  const orderTypes = ['buy_limit', 'buy_stop', 'sell_limit', 'sell_stop'];
  
  for (const orderType of orderTypes) {
    const percentage = await promptUser(`Notable percentage for ${orderType} (or press Enter to skip): `);
    if (percentage.trim() === '') continue;
    const value = await promptUser(`Notable value for ${orderType} (or press Enter to skip): `);
    notableOrders[orderType] = {
      percentage: parseFloat(percentage) || 0,
      value: parseFloat(value) || 0
    };
  }
  
  return Object.keys(notableOrders).length > 0 ? notableOrders : null;
}

// Function to collect all data
async function collectData() {
  const data = {};

  // Basic info
  data.currency_pair = await promptUser('Currency Pair (e.g., EUR/USD): ');
  data.current_timestamp = await promptUser('Current Timestamp (e.g., 2025-03-28T16:00:00Z, or press Enter for now): ') || new Date().toISOString();

  // Highest clusters
  data.highest_clusters = {};
  for (const type of ['buy_limit', 'buy_stop', 'sell_limit', 'sell_stop']) {
    const price = await promptUser(`Price of highest cluster for ${type} (or press Enter to skip): `);
    const percentage = await promptUser(`Percentage of highest cluster for ${type} (or press Enter to skip): `);
    data.highest_clusters[type] = {
      price: price.trim() === '' ? null : parseFloat(price) || 0,
      percentage: percentage.trim() === '' ? null : parseFloat(percentage) || 0
    };
  }

  // Open positions (profit/loss)
  data.open_positions = {};
  for (const type of ['profit', 'loss']) {
    const price = await promptUser(`Price of highest cluster for open positions ${type} (or press Enter to skip): `);
    const percentage = await promptUser(`Percentage of highest cluster for open positions ${type} (or press Enter to skip): `);
    data.open_positions[type] = {
      price: price.trim() === '' ? null : parseFloat(price) || 0,
      percentage: percentage.trim() === '' ? null : parseFloat(percentage) || 0
    };
  }

  // Price ranges where one order type dominates
  data.ranges = {};
  data.ranges.sell_limit_greater_than_buy_stop = await collectPriceRange('sell limit > buy stop');
  data.ranges.buy_stop_greater_than_sell_limit = await collectPriceRange('buy stop > sell limit');
  data.ranges.sell_stop_greater_than_buy_limit = await collectPriceRange('sell stop > buy limit');
  data.ranges.buy_limit_greater_than_sell_stop = await collectPriceRange('buy limit > sell stop');

  // Notable orders dictionary
  data.notable_orders = await collectNotableOrders();

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
  newEntry.entry_timestamp = new Date().toISOString();
  existingData.push(newEntry);

  // Write updated data back to file
  fs.writeFileSync(outputFile, JSON.stringify(existingData, null, 2), 'utf8');
  console.log(`Data saved to ${outputFile}. Total entries: ${existingData.length}`);
}

// Main execution
async function main() {
  try {
    console.log('Enter order and position data (press Ctrl+C to exit after saving).\n');
    
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