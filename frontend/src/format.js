// Small formatting and validation helpers (pure functions, unit-tested).

export function validateFile(file, info) {
  if (!file) return 'Choose an image first.';
  const formats = info?.formats ?? ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'];
  const dot = file.name.lastIndexOf('.');
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
  if (!formats.includes(ext)) {
    return `${file.name} is not a supported image type. Use ${formats.join(', ')}.`;
  }
  if (file.size === 0) return `${file.name} is empty.`;
  const maxMb = info?.limits?.max_file_mb;
  if (maxMb && file.size > maxMb * 1024 * 1024) {
    return `${file.name} is ${(file.size / 2 ** 20).toFixed(1)} MB; the limit is ${maxMb} MB.`;
  }
  return null;
}

export function validThreshold(v) {
  if (v === '' || v === null || v === undefined) return true;
  const n = Number(v);
  return Number.isFinite(n) && n >= 0 && n <= 1;
}

export const seconds = (ms) => (ms === null || ms === undefined ? '' : `${(ms / 1000).toFixed(2)} s`);

export function when(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export const STATUS_TEXT = { ok: 'Analysed', no_detections: 'No organisms', failed: 'Failed' };
export const DET_STATUS_TEXT = { organism: 'Confirmed', uncertain: 'Uncertain', reject: 'Rejected' };

// Rows of the counts table, in the same order and with the same totals as the server's reports.
export function countRows(result) {
  const c = result.counts;
  const e = result.estimated_counts;
  const reject = result.metadata.classes.reject;
  const rows = Object.keys(c.confirmed).map((k) => ({
    key: k, label: k, confirmed: c.confirmed[k], estimated: e.per_class[k], fromUncertain: e.from_uncertain[k],
  }));
  const orgFromUnc = Object.keys(c.confirmed).reduce((s, k) => s + e.from_uncertain[k], 0);
  return {
    rows,
    total: { key: 'all', label: 'All organisms', confirmed: c.confirmed_total, estimated: e.organisms_total, fromUncertain: orgFromUnc },
    rejected: { key: 'rejected', label: 'Rejected', confirmed: c.rejected, estimated: e.reject, fromUncertain: e.from_uncertain[reject] },
    uncertain: c.uncertain,
    detections: c.total_detections,
  };
}

// Sample calibration: the server's rule, checked before upload. Empty strings mean "not given".
// Returns null when the calibration is valid or entirely empty, else the problem in words.
export function calibrationError({ volumeMl = '', pixelUm = '', depthUm = '', factor = '' } = {}) {
  const given = { volumeMl, pixelUm, depthUm, factor };
  const has = (k) => given[k] !== '' && given[k] !== null && given[k] !== undefined;
  const filled = Object.keys(given).filter(has);
  if (filled.length === 0) return null;
  for (const k of filled) {
    const n = Number(given[k]);
    if (!Number.isFinite(n) || n <= 0) return 'Calibration values are positive numbers.';
  }
  if (has('volumeMl') && (has('pixelUm') || has('depthUm'))) return 'Give the imaged volume, or pixel size and depth, not both.';
  if (!has('volumeMl') && !(has('pixelUm') && has('depthUm'))) return 'Give the imaged volume, or both pixel size and depth.';
  return null;
}

// Three significant figures, never scientific notation, for the values a lab reads.
export function rate(x) {
  if (x === null || x === undefined || !Number.isFinite(x)) return '';
  if (x === 0) return '0';
  const digits = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(x))));
  return Number(x.toFixed(Math.min(digits, 8))).toLocaleString('en', { maximumFractionDigits: 8 });
}

// The 95% count range of one class ("0 to 2"), or '' for results saved before intervals existed.
export function countRange(result, key) {
  const v = result.count_intervals?.per_class?.[key];
  if (!v) return '';
  return v.low === v.high ? String(v.low) : `${v.low} to ${v.high}`;
}

// Long camera file names shortened in the middle, keeping the start and the extension ("SPC-EAWAG-0P5…-112.jpeg").
export function shortName(name, max = 28) {
  if (!name || name.length <= max) return name ?? '';
  const keepEnd = Math.min(12, Math.floor(max / 2));
  return `${name.slice(0, max - keepEnd - 1)}…${name.slice(-keepEnd)}`;
}
