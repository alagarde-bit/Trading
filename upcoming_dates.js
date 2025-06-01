const readline = require('readline');
const moment = require('moment');

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

// Editable event schedule (update dates here in 'YYYY-MM-DD' format)
const eventSchedule = {
  'US': {
    'Labor Market Reports': '2025-04-04', // Nonfarm Payrolls (example)
    'Inflationary Reports': '2025-04-10',  // CPI
    'Trade Balance Reports': '2025-04-08', // Trade Balance
    'Consumer Confidence': '2025-04-29',   // CB Consumer Confidence
    'Industrial Reports': '2025-04-17',    // Industrial Production
  },
  'Canada': {
    'Labor Market Reports': '2025-04-11',
    'Inflationary Reports': '2025-04-15',
    'Trade Balance Reports': '2025-04-09',
    'Consumer Confidence': '2025-04-22',
    'Industrial Reports': '2025-04-18',
  },
  'Japan': {
    'Labor Market Reports': '2025-04-28',
    'Inflationary Reports': '2025-04-25',
    'Trade Balance Reports': '2025-04-16',
    'Consumer Confidence': '2025-04-30',
    'Industrial Reports': '2025-04-23',
  },
  'UK': {
    'Labor Market Reports': '2025-04-15',
    'Inflationary Reports': '2025-04-16',
    'Trade Balance Reports': '2025-04-11',
    'Consumer Confidence': '2025-04-24',
    'Industrial Reports': '2025-04-21',
  },
  'Australia': {
    'Labor Market Reports': '2025-04-17',
    'Inflationary Reports': '2025-04-23',
    'Trade Balance Reports': '2025-04-03',
    'Consumer Confidence': '2025-04-08',
    'Industrial Reports': '2025-04-29',
  },
  'China': {
    'Labor Market Reports': '2025-04-14',
    'Inflationary Reports': '2025-04-10',
    'Trade Balance Reports': '2025-04-13',
    'Consumer Confidence': '2025-04-16',
    'Industrial Reports': '2025-04-20',
  },
  'Russia': {
    'Labor Market Reports': '2025-04-18',
    'Inflationary Reports': '2025-04-09',
    'Trade Balance Reports': '2025-04-15',
    'Consumer Confidence': '2025-04-22',
    'Industrial Reports': '2025-04-25',
  },
  'Switzerland': {
    'Labor Market Reports': '2025-04-30',
    'Inflationary Reports': '2025-04-03',
    'Trade Balance Reports': '2025-04-17',
    'Consumer Confidence': '2025-04-29',
    'Industrial Reports': '2025-04-10',
  },
  'New Zealand': {
    'Labor Market Reports': '2025-04-30',
    'Inflationary Reports': '2025-04-16',
    'Trade Balance Reports': '2025-04-23',
    'Consumer Confidence': '2025-04-25',
    'Industrial Reports': '2025-04-28',
  },
  'Slovakia': {
    'Labor Market Reports': '2025-04-11',
    'Inflationary Reports': '2025-04-14',
    'Trade Balance Reports': '2025-04-10',
    'Consumer Confidence': '2025-04-17',
    'Industrial Reports': '2025-04-20',
  },
  'Germany': {
    'Labor Market Reports': '2025-04-29',
    'Inflationary Reports': '2025-04-28',
    'Trade Balance Reports': '2025-04-08',
    'Consumer Confidence': '2025-04-24',
    'Industrial Reports': '2025-04-17',
  },
  'France': {
    'Labor Market Reports': '2025-04-10',
    'Inflationary Reports': '2025-04-15',
    'Trade Balance Reports': '2025-04-07',
    'Consumer Confidence': '2025-04-25',
    'Industrial Reports': '2025-04-22',
  },
};

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
    console.log(`\n${country}:`);
    eventTypes.forEach(event => {
      const nextDate = eventSchedule[country][event];
      const days = calculateDaysUntil(nextDate);
      console.log(`  ${event}: ${days} days (Next: ${nextDate})`);
    });
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
          eventSchedule[country][event] = date;
          console.log(`Updated ${country} - ${event} to ${date}`);
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