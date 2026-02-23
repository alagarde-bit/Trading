const fs = require('fs');
const readline = require('readline');
const moment = require('moment');

// Countries
const countries = [
  'US', 'Canada', 'Japan', 'UK', 'Australia', 'China',
  'Russia', 'Switzerland', 'New Zealand', 'Slovakia', 'Germany', 'France',
  'All', 'None'
];

// Trade event types (added Taxes and Internal Conflict)
const tradeEventTypes = [
  'Tariff', 'War', 'Sanction', 'Quota', 'Embargo',
  'More Demand for Import', 'More Demand for Export',
  'New Relation/Ally', 'New Disagreement/Enemy', 'Price Changes on Imports/Exports',
  'Taxes', 'Internal Conflict', 'Strategy'
];

// Data storage file
const DATA_FILE = 'trade_news.json';

// Load existing data or initialize
let tradeNews = {};
if (fs.existsSync(DATA_FILE)) {
  tradeNews = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
} else {
  countries.forEach(country => {
    tradeNews[country] = [];
  });
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function saveData() {
  fs.writeFileSync(DATA_FILE, JSON.stringify(tradeNews, null, 2));
}

function inputTradeNews() {
  rl.question('Enter country: ', country => {
    if (!countries.includes(country)) {
      console.log('Invalid country');
      rl.close();
      return;
    }
    rl.question('Enter other countries involved (comma-separated, e.g., Canada,Japan): ', others => {
      const otherCountries = others.split(',').map(c => c.trim());
      if (otherCountries.some(c => c && !countries.includes(c))) {
        console.log('Invalid country in "other countries"');
        rl.close();
        return;
      }
      rl.question('Enter trade event type: ', eventType => {
        if (!tradeEventTypes.includes(eventType)) {
          console.log('Invalid event type');
          rl.close();
          return;
        }
        rl.question('Enter brief description: ', description => {
          rl.question('Enter date (YYYY-MM-DD) or press Enter for today: ', date => {
            const inputDate = date ? moment(date, 'YYYY-MM-DD') : moment();
            if (date && !inputDate.isValid()) {
              console.log('Invalid date format');
              rl.close();
              return;
            }
            tradeNews[country].push({
              date: inputDate.format('YYYY-MM-DD'),
              otherCountries: otherCountries.filter(c => c), // Remove empty entries
              eventType,
              description,
            });
            saveData();
            console.log(`Added trade news for ${country} on ${inputDate.format('YYYY-MM-DD')}`);
            rl.close();
          });
        });
      });
    });
  });
}

function displayTradeNews() {
  rl.question('Enter country (or "all" for all countries): ', country => {
    if (country !== 'all' && !countries.includes(country)) {
      console.log('Invalid country');
      rl.close();
      return;
    }
    const targetCountries = country === 'all' ? countries : [country];

    targetCountries.forEach(c => {
      console.log(`\n${c}:`);
      tradeNews[c].forEach((entry, index) => {
        console.log(`  ${index + 1}. Date: ${entry.date}`);
        console.log(`     Other Countries: ${entry.otherCountries.join(', ') || 'None'}`);
        console.log(`     Event Type: ${entry.eventType}`);
        console.log(`     Description: ${entry.description}`);
      });
    });
    rl.close();
  });
}

// Main menu
console.log('1. Input new trade news');
console.log('2. Display trade news history');
rl.question('Choose option (1 or 2): ', choice => {
  if (choice === '1') {
    inputTradeNews();
  } else if (choice === '2') {
    displayTradeNews();
  } else {
    console.log('Invalid option');
    rl.close();
  }
});