const readline = require('readline');
const moment = require('moment');
const fs = require('fs');

// Economic event types
const eventTypes = [
  'Labor Market Reports',
  'Inflationary Reports',
  'Trade Balance Reports',
  'Consumer Confidence',
  'Industrial Reports',
];

// Countries
const countries = [
  'US', 'Canada', 'Japan', 'UK', 'Australia', 'China',
  'Russia', 'Switzerland', 'New Zealand', 'Slovakia', 'Germany', 'France',
];

// File paths
const scheduleFile = 'eventSchedule.json';
const historyFile = 'eventScheduleHistory.json';

// Load event schedule from file or initialize empty
let eventSchedule = {};
if (fs.existsSync(scheduleFile)) {
  try {
    eventSchedule = JSON.parse(fs.readFileSync(scheduleFile));
  } catch (err) {
    console.error('Error loading schedule:', err.message);
  }
}

// Function to save eventSchedule to file
function saveEventSchedule() {
  try {
    fs.writeFileSync(scheduleFile, JSON.stringify(eventSchedule, null, 2));
  } catch (err) {
    console.error('Error saving schedule:', err.message);
  }
}

// Function to append update to history file
function appendToHistory(country, event, date) {
  const timestamp = moment().format('YYYY-MM-DD HH:mm:ss');
  const historyEntry = { timestamp, country, event, date };
  let history = [];
  if (fs.existsSync(historyFile)) {
    try {
      history = JSON.parse(fs.readFileSync(historyFile));
    } catch (err) {
      console.error('Error reading history:', err.message);
    }
  }
  history.push(historyEntry);
  try {
    fs.writeFileSync(historyFile, JSON.stringify(history, null, 2));
  } catch (err) {
    console.error('Error saving history:', err.message);
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function calculateDaysUntil(nextDate) {
  const today = moment();
  const eventDate = moment(nextDate, 'YYYY-MM-DD');
  return eventDate.diff(today, 'days');
}

function displayEventDays() {
  console.log('\nDays until next economic events:');
  countries.forEach(country => {
    if (eventSchedule[country]) {
      console.log(`\n${country}:`);
      eventTypes.forEach(event => {
        const nextDate = eventSchedule[country][event];
        if (nextDate) {
          const days = calculateDaysUntil(nextDate);
          console.log(`  ${event}: ${days} days (Next: ${nextDate})`);
        } else {
          console.log(`  ${event}: Not scheduled`);
        }
      });
    }
  });
}

function updateEventDate() {
  rl.question('Enter country: ', country => {
    if (!countries.includes(country)) {
      console.log('Invalid country');
      rl.close();
      return;
    }
    rl.question('Enter event type: ', event => {
      if (!eventTypes.includes(event)) {
        console.log('Invalid event type');
        rl.close();
        return;
      }
      rl.question('Enter new date (YYYY-MM-DD): ', date => {
        if (!moment(date, 'YYYY-MM-DD', true).isValid()) {
          console.log('Invalid date format');
        } else {
          if (!eventSchedule[country]) {
            eventSchedule[country] = {};
          }
          eventSchedule[country][event] = date;
          console.log(`Updated ${country} - ${event} to ${date}`);
          saveEventSchedule();
          appendToHistory(country, event, date);
        }
        rl.close();
      });
    });
  });
}

// Main menu
console.log('1. Display days until next events');
console.log('2. Update event date');
rl.question('Choose option (1 or 2): ', choice => {
  if (choice === '1') {
    displayEventDays();
    rl.close();
  } else if (choice === '2') {
    updateEventDate();
  } else {
    console.log('Invalid option');
    rl.close();
  }
});