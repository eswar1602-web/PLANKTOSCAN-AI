import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../App.jsx';
import { INFO, LINKS, RESULT, json, mockFetch } from './fixtures.js';

const HEALTH = { status: 'ok', models_loaded: true, error: null, device: 'cuda' };
const base = {
  'GET /api/health': () => json(HEALTH),
  'GET /api/info': () => json(INFO),
  'GET /api/examples': () => json([{ name: 'val_00000.jpg', url: '/api/examples/val_00000.jpg', note: 'synthetic' }]),
  'GET /api/analyses': () => json({ items: [], total: 0, limit: 50, offset: 0 }),
};

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
