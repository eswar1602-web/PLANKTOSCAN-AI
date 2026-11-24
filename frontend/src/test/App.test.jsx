import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../App.jsx';
import { INFO, LINKS, RESULT, RESULT_CAL, RESULT_CONF, json, mockFetch } from './fixtures.js';

const HEALTH = { status: 'ok', models_loaded: true, error: null, device: 'cuda' };
const base = {
  'GET /api/health': () => json(HEALTH),
  'GET /api/info': () => json(INFO),
  'GET /api/examples': () => json([{ name: 'val_00000.jpg', url: '/api/examples/val_00000.jpg', note: 'synthetic' }]),
  'GET /api/analyses': () => json({ items: [], total: 0, limit: 50, offset: 0 }),
  'GET /api/corrections': () => json({ total: 0, analyses: 0, by_label: {}, agreed_with_model: 0 }),
  [`GET /api/analyses/${RESULT.analysis_id}/corrections`]: () => json(reviewView([])),
};

const ORG = RESULT.metadata.classes.organisms;
// The server's correction view for RESULT, computed the same way as src/corrections.py reviewed_counts.
function reviewView(items) {
  const by = Object.fromEntries(items.map((c) => [c.detection_id, c]));
  const confirmed = Object.fromEntries(ORG.map((k) => [k, 0]));
  let other = 0; let unc = 0; let agreed = 0;
  for (const d of RESULT.detections) {
    const c = by[d.id];
    if (!c) { if (d.status === 'organism') confirmed[d.label] += 1; else if (d.status === 'uncertain') unc += 1; continue; }
    if (c.corrected_label === d.label) agreed += 1;
    if (c.corrected_label in confirmed) confirmed[c.corrected_label] += 1;
    else if (c.corrected_label === 'other_organism') other += 1;
  }
  const total = Object.values(confirmed).reduce((a, b) => a + b, 0) + other;
  return {
    items, labels: [...ORG, 'reject', 'other_organism', 'not_an_object'],
    reviewed_counts: { confirmed, other_organism: other, organisms_total: total, unreviewed_uncertain: unc,
      reviewed: items.length, agreed, changed: items.length - agreed, detections: RESULT.detections.length },
  };
}

function setup(routes = {}) {
  const fetchMock = vi.fn(mockFetch({ ...base, ...routes }));
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();
  render(<App />);
  return { user, fetchMock };
}

afterEach(() => vi.unstubAllGlobals());

const png = () => new File([new Uint8Array([137, 80, 78, 71, 1, 2, 3])], 'frame.png', { type: 'image/png' });

describe('PlanktoScan app', () => {
  it('shows the model status and the empty analyse state', async () => {
    setup();
    expect(await screen.findByText(/Models ready on/)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Nothing analysed yet' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyse' })).toBeEnabled();
  });

  it('uploads an image and shows the four kinds of count, the boxes, detections and downloads', async () => {
    const { user, fetchMock } = setup({ 'POST /api/analyses': () => json({ result: RESULT, links: LINKS }, 201) });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));

    expect(await screen.findByRole('heading', { name: 'frame.png' })).toBeInTheDocument();
    const post = fetchMock.mock.calls.find(([, o]) => o?.method === 'POST');
    expect(post[1].body.get('file').name).toBe('frame.png');
    expect(post[1].body.get('det_conf')).toBeNull();                       // empty field -> server default

    const figures = screen.getByText('Confirmed organisms').closest('dl');
    expect(within(figures).getByText('Confirmed organisms').nextSibling).toHaveTextContent('1');
    expect(within(figures).getByText('Estimated organisms').nextSibling).toHaveTextContent('2');
    expect(within(figures).getByText('Uncertain').nextSibling).toHaveTextContent('1');
    expect(within(figures).getByText('Rejected').nextSibling).toHaveTextContent('1');
    expect(screen.getByText(RESULT.warnings[0])).toBeInTheDocument();
    expect(screen.getByRole('img', { name: '3 detection boxes' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Detections, 3' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'PDF report' })).toHaveAttribute('href', LINKS.reports.pdf);
    expect(screen.getByRole('link', { name: 'CSV' })).toHaveAttribute('href', LINKS.reports.csv);
    expect(screen.getAllByRole('link', { name: /report|CSV|JSON/ })).toHaveLength(4);
  });

  it('sends custom thresholds', async () => {
    const { user, fetchMock } = setup({ 'POST /api/analyses': () => json({ result: RESULT, links: LINKS }, 201) });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.type(screen.getByLabelText('Uncertainty'), '0.6');
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    await screen.findByRole('heading', { name: 'frame.png' });
    const post = fetchMock.mock.calls.find(([, o]) => o?.method === 'POST');
    expect(post[1].body.get('uncertainty')).toBe('0.6');
  });

  it('sends a calibration and shows ranges and concentrations', async () => {
    const { user, fetchMock } = setup({ 'POST /api/analyses': () => json({ result: RESULT_CAL, links: LINKS }, 201) });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByText('Sample calibration, for concentrations'));
    await user.type(screen.getByLabelText('Imaged volume, mL'), '0.002');
    await user.type(screen.getByLabelText('Concentration factor'), '50');
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    await screen.findByRole('heading', { name: 'frame.png' });
    const body = fetchMock.mock.calls.find(([, o]) => o?.method === 'POST')[1].body;
    expect(body.get('imaged_volume_ml')).toBe('0.002');
    expect(body.get('concentration_factor')).toBe('50');
    expect(body.get('pixel_size_um')).toBeNull();

    expect(screen.getByRole('columnheader', { name: '95% range' })).toBeInTheDocument();
    const conc = screen.getByRole('heading', { name: 'Concentration, per litre of original water' }).closest('section');
    const copepod = within(conc).getByRole('rowheader', { name: 'copepod' }).closest('tr');
    expect(copepod).toHaveTextContent('10,000');                    // 1 organism in 0.1 mL of water
    expect(conc).toHaveTextContent('concentrated 50 times');
  });

  it('shows calibrated probabilities, the unknown flag and the threshold sensitivity', async () => {
    const { user } = setup({ 'POST /api/analyses': () => json({ result: RESULT_CONF, links: LINKS }, 201) });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    await screen.findByRole('heading', { name: 'frame.png' });
    expect(screen.getByRole('columnheader', { name: 'Calibrated' })).toBeInTheDocument();
    expect(screen.getAllByText('Possibly unknown')).toHaveLength(1);
    expect(screen.getByText('0.610')).toBeInTheDocument();
    expect(screen.getByTestId('sensitivity')).toHaveTextContent('1, 1 and 1 confirmed organisms at uncertainty 0.88, 0.89 and 0.90');
    expect(screen.getByText(/1 detection looks unlike every trained class/)).toBeInTheDocument();
  });

  it('hides the calibrated column for results without calibration', async () => {
    const { user } = setup({ 'POST /api/analyses': () => json({ result: RESULT, links: LINKS }, 201) });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    await screen.findByRole('heading', { name: 'frame.png' });
    expect(screen.queryByRole('columnheader', { name: 'Calibrated' })).toBeNull();
    expect(screen.queryByTestId('sensitivity')).toBeNull();
  });

  it('saves a label correction and shows reviewed counts; clearing it deletes the correction', async () => {
    const det = `/api/analyses/${RESULT.analysis_id}/detections/2/correction`;
    const { user, fetchMock } = setup({
      'POST /api/analyses': () => json({ result: RESULT, links: LINKS }, 201),
      [`PUT ${det}`]: (_, o) => {
        const { label } = JSON.parse(o.body);
        const c = { detection_id: 2, corrected_label: label, predicted_label: 'rotifer', note: null };
        return json({ correction: c, ...reviewView([c]) });
      },
      [`DELETE ${det}`]: () => json(reviewView([])),
    });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    await screen.findByRole('heading', { name: 'frame.png' });
    expect(await screen.findByText(/set the right label in the last column/)).toBeInTheDocument();

    const select = screen.getByLabelText('Your label for detection 2');
    await waitFor(() => expect(select).toBeEnabled());
    expect(within(select).getByRole('option', { name: 'rotifer (as model)' })).toBeInTheDocument();
    await user.selectOptions(select, 'rotifer');
    const put = fetchMock.mock.calls.find(([, o]) => o?.method === 'PUT');
    expect(JSON.parse(put[1].body)).toEqual({ label: 'rotifer', note: null });
    expect(await screen.findByTestId('review-summary')).toHaveTextContent('Reviewed 1 of 3 detections: 1 agreed with the model, 0 changed.');
    const rotifer = within(screen.getByRole('heading', { name: 'Your review' }).closest('section'))
      .getByRole('rowheader', { name: 'rotifer' }).closest('tr');
    expect(rotifer).toHaveTextContent(/0\s*1/);                     // model 0 confirmed, 1 after review
    expect(await screen.findByText('Saved')).toBeInTheDocument();

    await user.selectOptions(select, '');
    await waitFor(() => expect(fetchMock.mock.calls.some(([, o]) => o?.method === 'DELETE')).toBe(true));
  });

  it('a late answer to an older save never undoes a newer correction', async () => {
    const id = RESULT.analysis_id;
    const c1 = { detection_id: 1, corrected_label: 'copepod', predicted_label: 'copepod', note: null };
    const c2 = { detection_id: 2, corrected_label: 'rotifer', predicted_label: 'rotifer', note: null };
    let release1;
    const slow1 = new Promise((r) => { release1 = r; });
    const { user } = setup({
      'POST /api/analyses': () => json({ result: RESULT, links: LINKS }, 201),
      // the server answered save 1 with a view read before save 2 was written, and that answer arrives last
      [`PUT /api/analyses/${id}/detections/1/correction`]: () => slow1.then(() => json({ correction: c1, ...reviewView([c1]) })),
      [`PUT /api/analyses/${id}/detections/2/correction`]: () => json({ correction: c2, ...reviewView([c1, c2]) }),
    });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    const s1 = await screen.findByLabelText('Your label for detection 1');
    const s2 = screen.getByLabelText('Your label for detection 2');
    await waitFor(() => expect(s1).toBeEnabled());
    await user.selectOptions(s1, 'copepod');
    await user.selectOptions(s2, 'rotifer');
    expect(await screen.findByTestId('review-summary')).toHaveTextContent('Reviewed 2 of 3');
    release1();
    await waitFor(() => expect(s1).toBeEnabled());
    expect(s2).toHaveValue('rotifer');
    expect(screen.getByTestId('review-summary')).toHaveTextContent('Reviewed 2 of 3');
  });

  it('shows a failed correction save on its row', async () => {
    const { user } = setup({
      'POST /api/analyses': () => json({ result: RESULT, links: LINKS }, 201),
      [`PUT /api/analyses/${RESULT.analysis_id}/detections/1/correction`]: () =>
        json({ error: { code: 'bad_request', message: 'Unknown label.' } }, 422),
    });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    const select = await screen.findByLabelText('Your label for detection 1');
    await waitFor(() => expect(select).toBeEnabled());
    await user.selectOptions(select, 'diatom');
    expect(await screen.findByText('Unknown label.')).toBeInTheDocument();
  });

  it('refuses an incomplete calibration in the browser', async () => {
    const { user, fetchMock } = setup();
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByText('Sample calibration, for concentrations'));
    await user.type(screen.getByLabelText('Pixel size, µm'), '1.2');
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    expect(await screen.findByText(/both pixel size and depth/)).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([, o]) => o?.method === 'POST')).toBe(false);
  });

  it('refuses an unsupported file in the browser without calling the server', async () => {
    const { fetchMock } = setup();
    const user = userEvent.setup({ applyAccept: false });   // as with drag and drop, where `accept` filters nothing
    await screen.findByText(/Models ready on/);
    const gif = new File(['GIF89a'], 'cells.gif', { type: 'image/gif' });
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), gif);
    expect(await screen.findByText(/cells.gif is not a supported image type/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    expect(fetchMock.mock.calls.some(([, o]) => o?.method === 'POST')).toBe(false);
  });

  it('shows the server error message and code', async () => {
    const { user } = setup({
      'POST /api/analyses': () => json({ error: { code: 'invalid_image', message: 'frame.png: the image data is corrupted or truncated.' } }, 400),
    });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('This image was not analysed');
    expect(alert).toHaveTextContent('corrupted or truncated');
    expect(alert).toHaveTextContent('invalid_image, HTTP 400');
  });

  it('explains an offline server', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))));
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Server not reachable' })).toBeInTheDocument();
    expect(screen.getByText(/python -m api/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyse' })).toBeDisabled();
  });

  it('disables analysis and says why when the models are missing', async () => {
    setup({ 'GET /api/health': () => json({ status: 'degraded', models_loaded: false, error: 'Model weights not found: models/x.pt.', device: null }) });
    expect(await screen.findByRole('heading', { name: 'Models not loaded' })).toBeInTheDocument();
    expect(screen.getByText(/Model weights not found/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyse' })).toBeDisabled();
  });

  it('history: empty state, then a list that opens a previous analysis', async () => {
    const { user } = setup();
    await user.click(screen.getByRole('link', { name: 'History' }));
    expect(await screen.findByRole('heading', { name: 'No analyses yet' })).toBeInTheDocument();
  });

  it('history shows reviewed counts per analysis and the corrections export', async () => {
    const item = { analysis_id: RESULT.analysis_id, created_at: RESULT.created_at, filename: 'frame.png', status: 'ok',
      confirmed_total: 1, estimated_total: 2, uncertain: 1, rejected: 1, processing_ms: 810, n_corrections: 2, links: LINKS };
    const { user } = setup({
      'GET /api/analyses': () => json({ items: [item], total: 1, limit: 50, offset: 0 }),
      'GET /api/corrections': () => json({ total: 2, analyses: 1, by_label: { rotifer: 2 }, agreed_with_model: 1 }),
    });
    await user.click(screen.getByRole('link', { name: 'History' }));
    const link = await screen.findByRole('link', { name: 'Export 2 corrections' });
    expect(link).toHaveAttribute('href', '/api/corrections/export');
    const row = screen.getByRole('link', { name: 'frame.png' }).closest('tr');
    expect(row.querySelector('[data-label="Reviewed"]')).toHaveTextContent('2');
  });

  it('opens and deletes a previous analysis', async () => {
    const item = { analysis_id: RESULT.analysis_id, created_at: RESULT.created_at, filename: 'frame.png', status: 'ok',
      confirmed_total: 1, estimated_total: 2, uncertain: 1, rejected: 1, processing_ms: 810, links: LINKS };
    let deleted = false;
    const { user } = setup({
      'GET /api/analyses': () => json({ items: deleted ? [] : [item], total: deleted ? 0 : 1, limit: 50, offset: 0 }),
      [`GET /api/analyses/${RESULT.analysis_id}`]: () => json({ ...item, warnings: [], result: RESULT, files_missing: false }),
      [`DELETE /api/analyses/${RESULT.analysis_id}`]: () => { deleted = true; return Promise.resolve(new Response(null, { status: 204 })); },
    });
    await user.click(screen.getByRole('link', { name: 'History' }));
    await user.click(await screen.findByRole('link', { name: 'frame.png' }));
    expect(await screen.findByRole('heading', { name: 'frame.png' })).toBeInTheDocument();
    expect(window.location.search).toContain(`analysis=${RESULT.analysis_id}`);
    await user.click(screen.getByRole('button', { name: 'Delete analysis' }));
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    expect(await screen.findByRole('heading', { name: 'No analyses yet' })).toBeInTheDocument();
  });

  it('shows a failed analysis from the history with its error', async () => {
    const failed = { analysis_id: 'f-1', created_at: RESULT.created_at, filename: 'bad.png', status: 'failed',
      error_code: 'invalid_image', error_message: 'bad.png: not a readable image.', links: { images: {}, reports: {} } };
    window.history.replaceState(null, '', '/?view=history&analysis=f-1');
    setup({ 'GET /api/analyses/f-1': () => json({ ...failed, warnings: [], result: null, files_missing: false }) });
    expect(await screen.findByRole('heading', { name: 'Not analysed' })).toBeInTheDocument();
    expect(screen.getByText('bad.png: not a readable image.')).toBeInTheDocument();
  });
});
