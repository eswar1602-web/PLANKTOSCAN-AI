import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.jsx';
import { resetLocalStore } from '../localStore.js';
import { INFO, RESULT, json, mockFetch } from './fixtures.js';

// A server in browser-storage mode (api/main.py, PLANKTOSCAN_STORAGE=browser): it keeps nothing, so the app keeps
// the history in the browser and asks the stateless endpoints for reports and reviewed counts.
const HEALTH = { status: 'ok', models_loaded: true, error: null, device: 'cpu', storage: 'browser' };
const ORG = RESULT.metadata.classes.organisms;
const IMG = { type: 'image/png', data: btoa('\x89PNG fake') };

function reviewView(corrections) {
  const items = corrections.map((c) => {
    const d = RESULT.detections.find((x) => x.id === c.detection_id);
    return { detection_id: c.detection_id, corrected_label: c.label, note: c.note ?? null,
      predicted_label: d.label, predicted_status: d.status, prob: d.prob, box: d.box };
  });
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

const routes = (extra = {}) => ({
  'GET /api/health': () => json(HEALTH),
  'GET /api/info': () => json({ ...INFO, storage: 'browser', limits: { ...INFO.limits, max_file_mb: 4 } }),
  'GET /api/examples': () => json([]),
  'POST /api/analyses': () => json({ result: RESULT, images: { annotated: IMG, enhanced: IMG }, review: reviewView([]), storage: 'browser' }, 201),
  'POST /api/review': (_, o) => json(reviewView(JSON.parse(o.body).corrections)),
  'POST /api/reports/pdf': () => Promise.resolve(new Response(new Blob(['%PDF-1.4']), {
    status: 200, headers: { 'content-type': 'application/pdf', 'content-disposition': 'attachment; filename="frame_x_planktoscan.pdf"' },
  })),
  ...extra,
});

let fetchMock;
function start(extra) {
  fetchMock = vi.fn(mockFetch(routes(extra)));
  vi.stubGlobal('fetch', fetchMock);
  render(<App />);
  return userEvent.setup();
}

const png = () => new File([new Uint8Array([137, 80, 78, 71, 1, 2, 3])], 'frame.png', { type: 'image/png' });
const historyCalls = () => fetchMock.mock.calls.filter(([u]) => /^\/api\/(analyses\/|analyses\?|corrections)/.test(String(u)));

async function analyse(user) {
  await screen.findByText(/Models ready on/);
  await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
  await user.click(screen.getByRole('button', { name: 'Analyse' }));
  await screen.findByRole('heading', { name: 'frame.png' });
}

beforeEach(() => resetLocalStore());
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('browser storage (stateless server)', () => {
  it('keeps the analysis in the browser: history lists it and opens it without asking the server', async () => {
    const user = await start();
    await analyse(user);
    expect(screen.getByRole('img', { name: 'frame.png, original view' })).toHaveAttribute('src', expect.stringMatching(/^blob:/));
    await user.click(screen.getByRole('link', { name: 'History' }));
    expect(await screen.findByTestId('local-history')).toHaveTextContent('kept in this browser only');
    expect(screen.queryByRole('link', { name: /Export/ })).not.toBeInTheDocument();
    const row = (await screen.findByRole('link', { name: 'frame.png' })).closest('tr');
    expect(row.querySelector('[data-label="Confirmed"]')).toHaveTextContent('1');
    await user.click(screen.getByRole('link', { name: 'frame.png' }));
    expect(await screen.findByRole('heading', { name: 'frame.png' })).toBeInTheDocument();
    expect(historyCalls()).toEqual([]);                                  // the server's history is never used
  });

  it('saves corrections through the stateless review and keeps them after a reload', async () => {
    const user = await start();
    await analyse(user);
    const select = screen.getByLabelText('Your label for detection 2');
    await waitFor(() => expect(select).toBeEnabled());
    await user.selectOptions(select, 'rotifer');
    expect(await screen.findByTestId('review-summary')).toHaveTextContent('Reviewed 1 of 3 detections: 1 agreed');
    await user.selectOptions(screen.getByLabelText('Your label for detection 3'), 'other_organism');
    await waitFor(() => expect(screen.getByTestId('review-summary')).toHaveTextContent('Reviewed 2 of 3 detections'));
    const last = fetchMock.mock.calls.filter(([u]) => u === '/api/review').at(-1);
    expect(JSON.parse(last[1].body).corrections).toEqual([
      { detection_id: 2, label: 'rotifer', note: null }, { detection_id: 3, label: 'other_organism', note: null }]);

    cleanup();                                                           // "reload" on the analysis's own link
    window.history.replaceState(null, '', `/?view=history&analysis=${RESULT.analysis_id}`);
    await start();
    expect(await screen.findByTestId('review-summary')).toHaveTextContent('Reviewed 2 of 3 detections');
    expect(screen.getByLabelText('Your label for detection 3')).toHaveValue('other_organism');
  });

  it('builds a report on demand and saves it under the server\'s file name', async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const user = await start();
    await analyse(user);
    await user.click(screen.getByRole('button', { name: 'PDF report' }));
    await waitFor(() => expect(click).toHaveBeenCalled());
    const [, opts] = fetchMock.mock.calls.find(([u]) => u === '/api/reports/pdf');
    expect(JSON.parse(opts.body.get('result')).analysis_id).toBe(RESULT.analysis_id);
    expect(opts.body.get('annotated')).toBeInstanceOf(Blob);
    expect(click.mock.contexts[0].download).toBe('frame_x_planktoscan.pdf');
  });

  it('shows a report error next to its button', async () => {
    const user = await start({
      'POST /api/reports/csv': () => json({ error: { code: 'bad_request', message: 'result: not a PlanktoScan analysis result.' } }, 422),
    });
    await analyse(user);
    await user.click(screen.getByRole('button', { name: 'CSV' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('not a PlanktoScan analysis result');
  });

  it('records a failed analysis in the browser history', async () => {
    const user = await start({
      'POST /api/analyses': () => json({ error: { code: 'invalid_image', message: 'frame.png: not a readable image.' } }, 400),
    });
    await screen.findByText(/Models ready on/);
    await user.upload(screen.getByLabelText(/Drop a microscopy image/), png());
    await user.click(screen.getByRole('button', { name: 'Analyse' }));
    expect(await screen.findByText('frame.png: not a readable image.')).toBeInTheDocument();
    await user.click(screen.getByRole('link', { name: 'History' }));
    const row = (await screen.findByRole('link', { name: 'frame.png' })).closest('tr');
    expect(within(row).getByText('Failed', { exact: false })).toBeInTheDocument();
  });

  it('deletes and clears the browser history', async () => {
    const user = await start();
    await analyse(user);
    await user.click(screen.getByRole('link', { name: 'History' }));
    await user.click(await screen.findByRole('link', { name: 'frame.png' }));
    await user.click(await screen.findByRole('button', { name: 'Delete analysis' }));
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    expect(await screen.findByRole('heading', { name: 'No analyses yet' })).toBeInTheDocument();
  });
});
