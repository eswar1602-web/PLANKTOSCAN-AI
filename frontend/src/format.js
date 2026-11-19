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
