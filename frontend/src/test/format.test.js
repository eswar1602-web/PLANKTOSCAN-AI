import { describe, expect, it } from 'vitest';
import { calibrationError, countRange, countRows, rate, seconds, shortName, validThreshold, validateFile } from '../format.js';
import { INFO, RESULT, RESULT_CAL } from './fixtures.js';

const file = (name, size) => ({ name, size });

describe('validateFile', () => {
  it('accepts supported images under the limit', () => {
    expect(validateFile(file('a.PNG', 10), INFO)).toBeNull();
    expect(validateFile(file('b.tiff', 10), INFO)).toBeNull();
  });
  it('rejects missing, unsupported, empty and oversized files with a usable message', () => {
    expect(validateFile(null, INFO)).toMatch(/Choose an image/);
    expect(validateFile(file('a.gif', 10), INFO)).toMatch(/not a supported image type/);
    expect(validateFile(file('noext', 10), INFO)).toMatch(/not a supported/);
    expect(validateFile(file('a.png', 0), INFO)).toMatch(/empty/);
    expect(validateFile(file('a.png', 51 * 2 ** 20), INFO)).toMatch(/limit is 50 MB/);
  });
});

describe('thresholds and formatting', () => {
  it('validThreshold allows empty (default) and 0..1 only', () => {
    expect(['', null, undefined, '0', '0.89', 1].every(validThreshold)).toBe(true);
    expect(['-0.1', '1.5', 'abc'].some(validThreshold)).toBe(false);
  });
  it('seconds formats milliseconds', () => {
    expect(seconds(1234)).toBe('1.23 s');
    expect(seconds(null)).toBe('');
  });
});

describe('countRows', () => {
  it('keeps confirmed, estimated, uncertain and rejected apart and adds up to the detections', () => {
    const c = countRows(RESULT);
    expect(c.rows).toHaveLength(7);
    expect(c.total).toMatchObject({ confirmed: 1, estimated: 2, fromUncertain: 1 });
    expect(c.rejected).toMatchObject({ confirmed: 1, estimated: 1 });
    expect(c.uncertain).toBe(1);
    expect(c.total.confirmed + c.uncertain + c.rejected.confirmed).toBe(c.detections);
    expect(c.total.estimated + c.rejected.estimated).toBe(c.detections);
  });
});

describe('calibration, rates and ranges', () => {
  it('accepts an empty calibration, a volume, or pixel size with depth', () => {
    expect(calibrationError({})).toBeNull();
    expect(calibrationError({ volumeMl: '0.002', factor: '50' })).toBeNull();
    expect(calibrationError({ pixelUm: '1.2', depthUm: '500' })).toBeNull();
  });
  it('refuses incomplete, mixed or non-positive calibrations', () => {
    expect(calibrationError({ pixelUm: '1.2' })).toMatch(/both pixel size and depth/);
    expect(calibrationError({ factor: '10' })).toMatch(/both pixel size and depth/);
    expect(calibrationError({ volumeMl: '1', depthUm: '5' })).toMatch(/not both/);
    expect(calibrationError({ volumeMl: '0' })).toMatch(/positive/);
    expect(calibrationError({ volumeMl: '-2' })).toMatch(/positive/);
  });
  it('rate keeps three significant figures', () => {
    expect(rate(10000)).toBe('10,000');
    expect(rate(36.8912)).toBe('36.9');
    expect(rate(0.0123456)).toBe('0.0123');
    expect(rate(0)).toBe('0');
    expect(rate(undefined)).toBe('');
  });
  it('countRange reads the interval, and is empty for results saved before intervals', () => {
    expect(countRange(RESULT_CAL, 'rotifer')).toBe('0 to 1');
    expect(countRange(RESULT_CAL, 'copepod')).toBe('1');
    expect(countRange(RESULT, 'copepod')).toBe('');
  });
});

describe('shortName', () => {
  it('keeps short names and shortens long ones in the middle, keeping the extension', () => {
    expect(shortName('val_00001.jpg')).toBe('val_00001.jpg');
    const long = 'SPC-EAWAG-0P5X-1561363726179016-8269122585331-005169-056-1368-1298-404-112.jpeg';
    const s = shortName(long);
    expect(s).toHaveLength(28);
    expect(s.startsWith('SPC-EAWAG-0P5')).toBe(true);
    expect(s.endsWith('404-112.jpeg')).toBe(true);
    expect(shortName(undefined)).toBe('');
  });
});
