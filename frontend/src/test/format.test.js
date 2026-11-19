import { describe, expect, it } from 'vitest';
import { countRows, seconds, validThreshold, validateFile } from '../format.js';
import { INFO, RESULT } from './fixtures.js';

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
