// REST client for the FastAPI backend (../api/main.py). Every failure becomes an ApiError with a code and a
// message a person can act on; the server's own {"error": {code, message}} envelope is passed through.

export class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

const OFFLINE =
  'Cannot reach the PlanktoScan server. Start it with "python -m api" (or "npm run dev" for development) and try again.';

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new ApiError('network_error', OFFLINE);
  }
  if (res.status === 204) return null;
  const type = res.headers.get('content-type') || '';
  const body = type.includes('application/json') ? await res.json().catch(() => null) : null;
  if (!res.ok) {
    const err = body && body.error;
    if (err) throw new ApiError(err.code, err.message, res.status);
    if (res.status === 502 || res.status === 504) throw new ApiError('network_error', OFFLINE, res.status);
    throw new ApiError('http_error', `The server answered ${res.status}. Try again, or check the server log.`, res.status);
  }
  return body;
}

export const getHealth = () => request('/api/health');
export const getInfo = () => request('/api/info');
export const listExamples = () => request('/api/examples');

const CALIBRATION_FIELDS = { volumeMl: 'imaged_volume_ml', pixelUm: 'pixel_size_um', depthUm: 'depth_um', factor: 'concentration_factor' };

export function analyze({ file, example, detConf, uncertainty, calibration }, signal) {
  const form = new FormData();
  if (file) form.append('file', file, file.name);
  else if (example) form.append('example', example);
  if (detConf !== undefined && detConf !== null && detConf !== '') form.append('det_conf', String(detConf));
  if (uncertainty !== undefined && uncertainty !== null && uncertainty !== '') form.append('uncertainty', String(uncertainty));
  for (const [k, field] of Object.entries(CALIBRATION_FIELDS)) {
    const v = calibration?.[k];
    if (v !== undefined && v !== null && v !== '') form.append(field, String(v));
  }
  return request('/api/analyses', { method: 'POST', body: form, signal });
}

export function listAnalyses({ limit = 50, offset = 0, status } = {}) {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) q.set('status', status);
  return request(`/api/analyses?${q}`);
}

export const getAnalysis = (id) => request(`/api/analyses/${encodeURIComponent(id)}`);
export const deleteAnalysis = (id) => request(`/api/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const clearHistory = () => request('/api/analyses?confirm=true', { method: 'DELETE' });

// Label corrections (one per detection; see src/corrections.py)
const detPath = (id, det) => `/api/analyses/${encodeURIComponent(id)}/detections/${det}/correction`;
export const getCorrections = (id) => request(`/api/analyses/${encodeURIComponent(id)}/corrections`);
export const putCorrection = (id, det, label, note = null) => request(detPath(id, det), {
  method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ label, note }),
});
export const deleteCorrection = (id, det) => request(detPath(id, det), { method: 'DELETE' });
export const correctionsSummary = () => request('/api/corrections');
export const CORRECTIONS_EXPORT_URL = '/api/corrections/export';
