// REST client for the FastAPI backend (../api/main.py). Every failure becomes an ApiError with a code and a
// message a person can act on; the server's own {"error": {code, message}} envelope is passed through.
//
// Two storage modes, chosen by the server (GET /api/health -> storage):
//   server   the server keeps the history, files and corrections (python -m api on one machine).
//   browser  stateless hosting (e.g. Vercel): the server keeps nothing, so the history lives in this browser
//            (localStore.js); reports and corrections are computed by stateless endpoints from what we send.
// The functions below have the same shapes in both modes, so the views never branch on it.

import { allRecords, clearRecords, deleteRecord, getRecord, listRecords, putRecord } from './localStore.js';

export class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

const OFFLINE =
  'Cannot reach the PlanktoScan server. An online copy that was idle takes about a minute to wake up: wait and try again. '
  + 'Running it yourself? Start it with "python -m api" (or "npm run dev" for development).';

let storage = 'server';
export const setStorage = (mode) => { storage = mode === 'browser' ? 'browser' : 'server'; };
export const browserStorage = () => storage === 'browser';

async function send(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError('network_error', OFFLINE);
  }
  if (!res.ok) {
    const type = res.headers.get('content-type') || '';
    const body = type.includes('application/json') ? await res.json().catch(() => null) : null;
    const err = body && body.error;
    if (err) throw new ApiError(err.code, err.message, res.status);
    if (res.status === 502 || res.status === 504) throw new ApiError('network_error', OFFLINE, res.status);
    if (res.status === 413) throw new ApiError('image_too_large', 'The file is too large for this server.', 413);
    throw new ApiError('http_error', `The server answered ${res.status}. Try again, or check the server log.`, res.status);
  }
  return res;
}

async function request(path, options = {}) {
  const res = await send(path, options);
  if (res.status === 204) return null;
  const type = res.headers.get('content-type') || '';
  return type.includes('application/json') ? res.json().catch(() => null) : null;
}

const postJson = (path, body) => request(path, {
  method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
});

export const getHealth = () => request('/api/health');
export const getInfo = () => request('/api/info');
export const listExamples = () => request('/api/examples');

const CALIBRATION_FIELDS = { volumeMl: 'imaged_volume_ml', pixelUm: 'pixel_size_um', depthUm: 'depth_um', factor: 'concentration_factor' };

function analysisForm({ file, example, detConf, uncertainty, calibration }) {
  const form = new FormData();
  if (file) form.append('file', file, file.name);
  else if (example) form.append('example', example);
  if (detConf !== undefined && detConf !== null && detConf !== '') form.append('det_conf', String(detConf));
  if (uncertainty !== undefined && uncertainty !== null && uncertainty !== '') form.append('uncertainty', String(uncertainty));
  for (const [k, field] of Object.entries(CALIBRATION_FIELDS)) {
    const v = calibration?.[k];
    if (v !== undefined && v !== null && v !== '') form.append(field, String(v));
  }
  return form;
}

export function analyze(args, signal) {
  if (browserStorage()) return analyzeInBrowser(args, signal);
  return request('/api/analyses', { method: 'POST', body: analysisForm(args), signal });
}

export function listAnalyses({ limit = 50, offset = 0, status } = {}) {
  if (browserStorage()) return listLocal({ limit, offset, status });
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) q.set('status', status);
  return request(`/api/analyses?${q}`);
}

export const getAnalysis = (id) => (browserStorage() ? getLocal(id)
  : request(`/api/analyses/${encodeURIComponent(id)}`));
export const deleteAnalysis = (id) => (browserStorage() ? deleteLocal(id)
  : request(`/api/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' }));
export const clearHistory = () => (browserStorage() ? clearLocal()
  : request('/api/analyses?confirm=true', { method: 'DELETE' }));

// Label corrections (one per detection; see src/corrections.py)
const detPath = (id, det) => `/api/analyses/${encodeURIComponent(id)}/detections/${det}/correction`;
export const getCorrections = (id) => (browserStorage() ? localReview(id)
  : request(`/api/analyses/${encodeURIComponent(id)}/corrections`));
export const putCorrection = (id, det, label, note = null) => (browserStorage() ? setLocalCorrection(id, det, label, note)
  : request(detPath(id, det), {
    method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ label, note }),
  }));
export const deleteCorrection = (id, det) => (browserStorage() ? setLocalCorrection(id, det, null)
  : request(detPath(id, det), { method: 'DELETE' }));
export const correctionsSummary = () => (browserStorage() ? localCorrectionsSummary() : request('/api/corrections'));
export const CORRECTIONS_EXPORT_URL = '/api/corrections/export';

// ---------------------------------------------------------------- browser storage
// Failures the server would have kept in its history (validation and inference errors of one image).
const RECORDED = new Set(['invalid_image', 'unsupported_format', 'image_too_small', 'image_too_large', 'inference_error']);
const REPORT_FORMATS = ['csv', 'json', 'html', 'pdf'];
const urls = new Map(); // analysis id -> {kind: object URL}, so each image is turned into a URL once

function b64Blob({ type, data }) {
  const bin = atob(data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

function localId() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  const rand = Math.random().toString(16).slice(2, 10).padEnd(8, '0');
  return `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}-${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}-${rand}`;
}

async function analyzeInBrowser(args, signal) {
  const name = args.file ? args.file.name : args.example;
  const t0 = performance.now();
  let body;
  try {
    body = await request('/api/analyses', { method: 'POST', body: analysisForm(args), signal });
  } catch (err) {
    if (err instanceof ApiError && RECORDED.has(err.code)) {
      await putRecord({
        analysis_id: localId(), created_at: new Date().toISOString(), filename: name, status: 'failed',
        error_code: err.code, error_message: err.message, processing_ms: performance.now() - t0, result: null, files: {},
      }).catch(() => {});
    }
    throw err;
  }
  let original = args.file || null;
  if (!original && args.example) {
    original = await fetch(`/api/examples/${encodeURIComponent(args.example)}`)
      .then((r) => (r.ok ? r.blob() : null)).catch(() => null);
  }
  const { result } = body;
  const rec = {
    analysis_id: result.analysis_id, created_at: result.created_at, filename: result.filename, status: result.status,
    error_code: null, error_message: null, processing_ms: result.timings_ms?.total ?? null, result,
    files: { original, annotated: b64Blob(body.images.annotated), enhanced: b64Blob(body.images.enhanced) },
    review: body.review,
  };
  try {
    await putRecord(rec);
  } catch {
    rec.result = { ...result, warnings: [...result.warnings, 'This analysis could not be kept in the browser history (storage full or blocked).'] };
  }
  return { result: rec.result, links: links(rec) };
}

function links(rec) {
  if (!urls.has(rec.analysis_id)) {
    const made = {};
    for (const [k, blob] of Object.entries(rec.files || {})) if (blob) made[k] = URL.createObjectURL(blob);
    urls.set(rec.analysis_id, made);
  }
  const images = urls.get(rec.analysis_id);
  const reports = rec.result ? Object.fromEntries(REPORT_FORMATS.map((f) => [f, () => buildReport(rec, f)])) : {};
  return { images: { ...images }, reports };
}

function forgetUrls(id) {
  for (const u of Object.values(urls.get(id) || {})) URL.revokeObjectURL(u);
  urls.delete(id);
}

// A report built by the server from the stored result and images: resolves to {blob, filename}.
async function buildReport(rec, fmt) {
  const form = new FormData();
  form.append('result', JSON.stringify(rec.result));
  if ((fmt === 'html' || fmt === 'pdf') && rec.files.annotated) form.append('annotated', rec.files.annotated, 'annotated');
  if (fmt === 'html' && rec.files.enhanced) form.append('enhanced', rec.files.enhanced, 'enhanced');
  const res = await send(`/api/reports/${fmt}`, { method: 'POST', body: form });
  const m = /filename="([^"]+)"/.exec(res.headers.get('content-disposition') || '');
  return { blob: await res.blob(), filename: m ? m[1] : `${rec.analysis_id}_planktoscan.${fmt}` };
}

function summary(rec) {
  const r = rec.result;
  return {
    analysis_id: rec.analysis_id, created_at: rec.created_at, filename: rec.filename, status: rec.status,
    error_code: rec.error_code ?? null, error_message: rec.error_message ?? null,
    n_detections: r ? r.counts.total_detections : null, confirmed_total: r ? r.counts.confirmed_total : null,
    estimated_total: r ? r.estimated_counts.organisms_total : null, uncertain: r ? r.counts.uncertain : null,
    rejected: r ? r.counts.rejected : null, processing_ms: rec.processing_ms ?? null,
    det_conf: r ? r.metadata.thresholds.det_conf : null, uncertainty: r ? r.metadata.thresholds.uncertainty : null,
    classifier_run: r ? r.metadata.models?.classifier?.train_run ?? null : null,
    n_corrections: rec.review?.items?.length ?? 0, links: links(rec),
  };
}

const notFound = (id) => new ApiError('not_found', `No analysis with id ${id} in this browser's history.`, 404);

async function listLocal({ limit, offset, status }) {
  const { items, total } = await listRecords({ limit, offset, status });
  return { items: items.map(summary), total, limit, offset };
}

async function getLocal(id) {
  const rec = await getRecord(id);
  if (!rec) throw notFound(id);
  return { ...summary(rec), warnings: rec.result?.warnings ?? [], result: rec.result, files_missing: false };
}

async function deleteLocal(id) {
  if (!(await deleteRecord(id))) throw notFound(id);
  forgetUrls(id);
  return null;
}

async function clearLocal() {
  for (const id of [...urls.keys()]) forgetUrls(id);
  return { deleted: await clearRecords() };
}

async function reviewable(id) {
  const rec = await getRecord(id);
  if (!rec) throw notFound(id);
  if (!rec.result) throw new ApiError('no_result', `Analysis ${id} has no stored result to correct.`, 409);
  return rec;
}

async function localReview(id) {
  const rec = await reviewable(id);
  return rec.review ?? postJson('/api/review', { result: rec.result, corrections: [] });
}

// Saves on one analysis run one after another: each reads the record the previous one wrote.
const queues = new Map();
function serial(id, fn) {
  const next = (queues.get(id) || Promise.resolve()).catch(() => {}).then(fn);
  queues.set(id, next);
  return next;
}

function setLocalCorrection(id, det, label, note = null) {
  return serial(id, async () => {
    const rec = await reviewable(id);
    const current = rec.review?.items ?? [];
    if (!label && !current.some((c) => c.detection_id === det)) {
      throw new ApiError('not_found', `Detection ${det} of ${id} has no correction.`, 404);
    }
    const items = current.filter((c) => c.detection_id !== det)
      .map((c) => ({ detection_id: c.detection_id, label: c.corrected_label, note: c.note ?? null }));
    if (label) items.push({ detection_id: det, label, note });
    const view = await postJson('/api/review', { result: rec.result, corrections: items });
    await putRecord({ ...rec, review: view });
    return label ? { correction: view.items.find((c) => c.detection_id === det), ...view } : view;
  });
}

async function localCorrectionsSummary() {
  const records = await allRecords();
  const rows = records.flatMap((r) => r.review?.items ?? []);
  const byLabel = {};
  for (const r of rows) byLabel[r.corrected_label] = (byLabel[r.corrected_label] || 0) + 1;
  return {
    total: rows.length, analyses: records.filter((r) => r.review?.items?.length).length, by_label: byLabel,
    agreed_with_model: rows.filter((r) => r.corrected_label === r.predicted_label).length,
  };
}

// Save a downloaded Blob under a name.
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
