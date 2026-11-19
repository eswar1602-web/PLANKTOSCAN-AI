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

export function analyze({ file, example, detConf, uncertainty }, signal) {
  const form = new FormData();
  if (file) form.append('file', file, file.name);
  else if (example) form.append('example', example);
  if (detConf !== undefined && detConf !== null && detConf !== '') form.append('det_conf', String(detConf));
  if (uncertainty !== undefined && uncertainty !== null && uncertainty !== '') form.append('uncertainty', String(uncertainty));
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
