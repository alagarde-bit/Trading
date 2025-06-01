const fs = require('fs');
const { parse } = require('csv-parse');
const { pipeline } = require('stream');

// Install required package: npm install csv-parse

// List of countries and their associated currencies
const countries = [
  'US', 'Canada', 'Japan', 'UK', 'Australia', 'China',
  'Russia', 'Switzerland', 'New Zealand', 'Slovakia', 'Germany', 'France'
];
const currencyMap = {
  'US': 'USD',
  'Canada': 'CAD',
  'Japan': 'JPY',
  'UK': 'GBP',
  'Australia': 'AUD',
  'China': 'CNY',
  'Russia': 'RUB',
  'Switzerland': 'CHF',
  'New Zealand': 'NZD',
  'Slovakia': 'EUR',
  'Germany': 'EUR',
  'France': 'EUR'
};

// Currency keywords to filter
const currencyKeywords = [
  'CANADIAN DOLLAR', 'JAPANESE YEN', 'BRITISH POUND', 'AUSTRALIAN DOLLAR',
  'SWISS FRANC', 'NEW ZEALAND DOLLAR', 'EURO FX', 'USD INDEX'
];

// Output file
const outputFile = 'asset_manager_currency_data.json';

// Function to process CSV stream and extract data
function processCSV(inputFile) {
  const results = [];
  
  return new Promise((resolve, reject) => {
    pipeline(
      fs.createReadStream(inputFile),
      parse({ columns: true, trim: true }),
      async function* (source) {
        for await (const row of source) {
          const marketName = row['Market_and_Exchange_Names'].toUpperCase();
          if (currencyKeywords.some(keyword => marketName.includes(keyword))) {
            results.push({
              Market_and_Exchange_Names: row['Market_and_Exchange_Names'],
              Report_Date_as_YYYY_MM_DD: row['Report_Date_as_YYYY_MM_DD'],
              Asset_Mgr_Positions_Long_All: parseInt(row['Asset_Mgr_Positions_Long_All']) || 0,
              Asset_Mgr_Positions_Short_All: parseInt(row['Asset_Mgr_Positions_Short_All']) || 0,
              Change_in_Asset_Mgr_Long_All: parseInt(row['Change_in_Asset_Mgr_Long_All']) || 0,
              Change_in_Asset_Mgr_Short_All: parseInt(row['Change_in_Asset_Mgr_Short_All']) || 0,
              Net_Position: (parseInt(row['Asset_Mgr_Positions_Long_All']) || 0) - (parseInt(row['Asset_Mgr_Positions_Short_All']) || 0)
            });
          }
        }
      },
      (err) => {
        if (err) reject(err);
        else resolve(results);
      }
    );
  });
}

// Function to append or update data in JSON file
async function updateJSONFile(newData) {
  let existingData = [];
  
  // Check if output file exists and load it
  if (fs.existsSync(outputFile)) {
    const fileContent = fs.readFileSync(outputFile, 'utf8');
    existingData = JSON.parse(fileContent || '[]');
  }

  // Filter out duplicates based on Market_and_Exchange_Names and Report_Date
  const updatedData = [
    ...existingData.filter(existing => 
      !newData.some(newEntry => 
        newEntry.Market_and_Exchange_Names === existing.Market_and_Exchange_Names &&
        newEntry.Report_Date_as_YYYY_MM_DD === existing.Report_Date_as_YYYY_MM_DD
      )
    ),
    ...newData
  ];

  // Write updated data back to file
  fs.writeFileSync(outputFile, JSON.stringify(updatedData, null, 2), 'utf8');
  console.log(`Data updated in ${outputFile}. Total entries: ${updatedData.length}`);
}

// Main execution
async function main() {
  try {
    const inputFile = 'weekly_data.csv'; // Replace with your weekly CSV file path
    const newData = await processCSV(inputFile);
    await updateJSONFile(newData);
  } catch (error) {
    console.error('Error processing data:', error);
  }
}

main();