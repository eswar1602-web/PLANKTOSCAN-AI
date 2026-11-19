// A canonical result shaped exactly like AnalysisResult.to_dict() from src/service.py.
const ORGS = ['copepod', 'diatom', 'nauplius', 'cladoceran', 'rotifer', 'ceratium', 'dinobryon'];
const zero = (extra = {}) => ({ ...Object.fromEntries(ORGS.map((k) => [k, 0])), ...extra });

export const RESULT = {
  analysis_id: '20261120-101500-abcd1234',
  created_at: '2026-11-20T10:15:00.000000+00:00',
  filename: 'frame.png',
  source_path: null,
  status: 'ok',
  message: '3 objects detected.',
  counts: { confirmed: zero({ copepod: 1 }), confirmed_total: 1, uncertain: 1, rejected: 1, total_detections: 3 },
  estimated_counts: {
    per_class: zero({ copepod: 1, rotifer: 1 }), organisms_total: 2, reject: 1,
    from_uncertain: { ...zero({ rotifer: 1 }), reject: 0 }, total: 3,
  },
  detections: [
    { id: 1, status: 'organism', label: 'copepod', prob: 0.97, det_conf: 0.91, box: { x0: 10, y0: 10, x1: 50, y1: 60 }, probs: {} },
    { id: 2, status: 'uncertain', label: 'rotifer', prob: 0.55, det_conf: 0.8, box: { x0: 70, y0: 20, x1: 110, y1: 70 }, probs: {} },
    { id: 3, status: 'reject', label: 'reject', prob: 0.99, det_conf: 0.7, box: { x0: 120, y0: 30, x1: 150, y1: 60 }, probs: {} },
  ],
  timings_ms: { load_and_validate: 4, detection: 40, classification: 60, counting: 0.1, annotation: 2, enhancement_preview: 700, total: 810 },
  warnings: ['1 of 3 detections are uncertain: the image may be outside the training domain.'],
  metadata: {
    image: { width: 160, height: 120 },
    thresholds: { det_conf: 0.5, uncertainty: 0.89, nms_iou: 0.75, defaults: { det_conf: 0.5, uncertainty: 0.89 } },
    classes: { organisms: ORGS, reject: 'reject' },
    models: { device: 'cpu', classifier: { file: 'classifier_best.pt', train_run: 'run' }, detector: { file: 'detector_best.pt' } },
    disclaimer: 'Results on real microscope images are NOT validated.',
  },
};

const base = `/api/analyses/${RESULT.analysis_id}`;
export const LINKS = {
  images: { original: `${base}/images/original`, enhanced: `${base}/images/enhanced`, annotated: `${base}/images/annotated` },
  reports: { csv: `${base}/reports/csv`, json: `${base}/reports/json`, html: `${base}/reports/html`, pdf: `${base}/reports/pdf` },
};

export const INFO = {
  defaults: { det_conf: 0.5, uncertainty: 0.89, nms_iou: 0.75 },
  formats: ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],
  limits: { max_file_mb: 50, max_megapixels: 40, min_side_px: 32 },
  classes: { organisms: ORGS, reject: 'reject' },
  models: RESULT.metadata.models,
  disclaimer: 'NOT validated',
};

export function json(body, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } }));
}

// A fetch mock routing by method + path; unknown routes answer 404 in the API's error envelope.
export function mockFetch(routes) {
  return (url, opts = {}) => {
    const path = String(url);
    const key = `${(opts.method || 'GET').toUpperCase()} ${path.split('?')[0]}`;
    const handler = routes[key];
    if (!handler) return json({ error: { code: 'not_found', message: `no mock for ${key}` } }, 404);
    return handler(path, opts);
  };
}
