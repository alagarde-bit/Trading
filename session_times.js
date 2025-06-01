const moment = require('moment-timezone');

// Session locations with timezones and peak volume times (UTC)
const sessionData = {
  NY: { timezone: 'America/New_York', peakStart: '13:00', peakEnd: '17:00' },
  London: { timezone: 'Europe/London', peakStart: '13:00', peakEnd: '17:00' },
  Tokyo: { timezone: 'Asia/Tokyo', peakStart: '03:00', peakEnd: '06:00' },
  Sydney: { timezone: 'Australia/Sydney', peakStart: '00:00', peakEnd: '03:00' },
  HK: { timezone: 'Asia/Hong_Kong', peakStart: '04:00', peakEnd: '05:00' },
  Singapore: { timezone: 'Asia/Singapore', peakStart: '04:00', peakEnd: '05:00' },
  Frankfurt: { timezone: 'Europe/Berlin', peakStart: '11:00', peakEnd: '12:00' },
};

function getSessionTimes() {
  const now = moment.utc(); // Current UTC time at trade decision
  const currentTime = now.format('HH:mm');
  const sessionTimes = {};

  for (const [session, { timezone, peakStart, peakEnd }] of Object.entries(sessionData)) {
    const localTime = moment(now).tz(timezone).format('YYYY-MM-DD HH:mm z');
    const isPeak = currentTime >= peakStart && currentTime < peakEnd; // Boolean for peak volume time
    sessionTimes[session] = {
      time: localTime,
      inPeakVolume: isPeak,
    };
  }

  return sessionTimes;
}

// Example usage
const times = getSessionTimes();
console.log(`Times at trade decision (${moment.utc().format('YYYY-MM-DD HH:mm')} UTC):`);
console.log(times);