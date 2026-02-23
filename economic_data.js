const fs = require('fs');
const readline = require('readline');
const moment = require('moment');

// Countries
const countries = [
  'US', 'Canada', 'Japan', 'UK', 'Australia', 'China',
  'Russia', 'Switzerland', 'New Zealand', 'Slovakia', 'Germany', 'France',
];

// Economic indicators
const indicators = [
  'Interest Rate', 'Inflation Rate', 'Retail Price Index', 'Balance of Trade',
  'Unemployment Rate', 'GDP Growth Rate', 'Manufacturing PMI', 'Non Farm Payrolls',
  'CPI', 'Consumer Confidence Index', 'Home Sales', '10-Year Bond Yields'
];

// Data storage file
const DATA_FILE = 'economic_data.json';

// Load existing data or initialize
let economicData = {};
if (fs.existsSync(DATA_FILE)) {
  economicData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
} else {
  countries.forEach(country => {
    economicData[country] = {};
    indicators.forEach(indicator => {
      economicData[country][indicator] = [];
    });
  });
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function saveData() {
  fs.writeFileSync(DATA_FILE, JSON.stringify(economicData, null, 2));
}

function inputData() {
  rl.question('Enter country: ', country => {
    if (!countries.includes(country)) {
      console.log('Invalid country');
      rl.close();
      return;
    }
    rl.question('Enter indicator: ', indicator => {
      if (!indicators.includes(indicator)) {
        console.log('Invalid indicator');
        rl.close();
        return;
      }
      rl.question('Enter value: ', value => {
        if (isNaN(value) && indicator !== 'Balance of Trade' && indicator !== 'Non Farm Payrolls') {
          console.log('Value must be a number');
          rl.close();
          return;
        }
        rl.question('Enter date (YYYY-MM-DD) or press Enter for today: ', date => {
          const inputDate = date ? moment(date, 'YYYY-MM-DD') : moment();
          if (date && !inputDate.isValid()) {
            console.log('Invalid date format');
            rl.close();
            return;
          }
          economicData[country][indicator].push({
            date: inputDate.format('YYYY-MM-DD'),
            value: parseFloat(value) || value, // Keep as string for Balance of Trade, Non Farm Payrolls
          });
          saveData();
          console.log(`Added ${value} to ${country} - ${indicator} on ${inputDate.format('YYYY-MM-DD')}`);
          rl.close();
        });
      });
    });
  });
}

function displayHistory() {
  rl.question('Enter country (or "all" for all countries): ', country => {
    if (country !== 'all' && !countries.includes(country)) {
      console.log('Invalid country');
      rl.close();
      return;
    }
    rl.question('Enter indicator (or "all" for all indicators): ', indicator => {
      if (indicator !== 'all' && !indicators.includes(indicator)) {
        console.log('Invalid indicator');
        rl.close();
        return;
      }
      const targetCountries = country === 'all' ? countries : [country];
      const targetIndicators = indicator === 'all' ? indicators : [indicator];

      targetCountries.forEach(c => {
        console.log(`\n${c}:`);
        targetIndicators.forEach(i => {
          console.log(`  ${i}:`);
          economicData[c][i].forEach(entry => {
            console.log(`    ${entry.date}: ${entry.value}`);
          });
        });
      });
      rl.close();
    });
  });
}

// Main menu
console.log('1. Input new data');
console.log('2. Display history');
rl.question('Choose option (1 or 2): ', choice => {
  if (choice === '1') {
    inputData();
  } else if (choice === '2') {
    displayHistory();
  } else {
    console.log('Invalid option');
    rl.close();
  }
});