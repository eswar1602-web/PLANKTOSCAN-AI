import { useCallback, useEffect, useState } from 'react';
import { clearHistory, listAnalyses } from '../api.js';
import { STATUS_TEXT, seconds, when } from '../format.js';

const PAGE = 50;

export default function HistoryView({ onOpen, refreshKey }) {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [state, setState] = useState('loading'); // loading | ready | error
  const [error, setError] = useState(null);
  const [more, setMore] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [clearError, setClearError] = useState(null);

  const load = useCallback(async () => {
    setState('loading');
    try {
      const page = await listAnalyses({ limit: PAGE });
      setItems(page.items);
      setTotal(page.total);
      setState('ready');
    } catch (err) {
      setError(err);
      setState('error');
    }
  }, []);

  useEffect(() => { load(); }, [load, refreshKey]);

  async function loadMore() {
    setMore(true);
    try {
      const page = await listAnalyses({ limit: PAGE, offset: items.length });
      setItems((xs) => [...xs, ...page.items]);
      setTotal(page.total);
    } catch (err) {
      setError(err);
    } finally {
      setMore(false);
    }
  }

  async function clearAll() {
    try {
      await clearHistory();
      setConfirming(false);
      setClearError(null);
      load();
    } catch (err) {
      setClearError(err.message);
    }
  }

  return (
    <section className="history" aria-labelledby="history-title">
      <header className="history__head">
        <h2 id="history-title">Analysis history</h2>
        {state === 'ready' && total > 0 && (
          <div className="history__actions">
            {confirming ? (
              <>
                <span className="help">Delete all {total} analyses and their stored files?</span>
                <button type="button" className="btn btn--danger" onClick={clearAll}>Delete all</button>
                <button type="button" className="btn btn--outline" onClick={() => setConfirming(false)}>Keep them</button>
              </>
            ) : (
              <button type="button" className="btn btn--outline" onClick={() => setConfirming(true)}>Clear history</button>
            )}
          </div>
        )}
      </header>
      {clearError && <p className="alert" role="alert">{clearError}</p>}

      {state === 'loading' && (
        <div className="skeleton skeleton--table" role="status" aria-label="Loading the history">
          {Array.from({ length: 5 }, (_, i) => <span key={i} />)}
        </div>
      )}
      {state === 'error' && (
        <div className="alert" role="alert">
          <h3>The history could not be loaded</h3>
          <p>{error?.message}</p>
          <button type="button" className="btn btn--outline" onClick={load}>Try again</button>
        </div>
      )}
      {state === 'ready' && items.length === 0 && (
        <div className="empty-state">
          <h3>No analyses yet</h3>
          <p>Every image you analyse is kept here with its counts, images and reports.</p>
          <a className="btn btn--primary" href="?view=analyze" onClick={(e) => { e.preventDefault(); onOpen(null); }}>Analyse an image</a>
        </div>
      )}
      {state === 'ready' && items.length > 0 && (
        <>
          <div className="table-scroll" tabIndex={0} aria-label="History table, scrolls sideways">
            <table className="spec spec--rows spec--history">
              <thead>
                <tr>
                  <th scope="col">Image</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="n">Confirmed</th>
                  <th scope="col" className="n">Estimated</th>
                  <th scope="col" className="n">Uncertain</th>
                  <th scope="col" className="n">Rejected</th>
                  <th scope="col" className="n">Time</th>
                  <th scope="col">Analysed</th>
                  <th scope="col">Id</th>
                </tr>
              </thead>
              <tbody>
                {items.map((r) => (
                  <tr key={r.analysis_id}>
                    <th scope="row">
                      <a className="link" href={`?view=history&analysis=${r.analysis_id}`}
                        onClick={(e) => { e.preventDefault(); onOpen(r.analysis_id); }}>
                        {r.filename}
                      </a>
                      {r.status === 'failed' && <span className="row-error">{r.error_message}</span>}
                    </th>
                    <td data-label="Status"><span className={`status status--${r.status}`}>{STATUS_TEXT[r.status]}</span></td>
                    <td className="n code" data-label="Confirmed">{r.confirmed_total ?? ''}</td>
                    <td className="n code" data-label="Estimated">{r.estimated_total ?? ''}</td>
                    <td className="n code" data-label="Uncertain">{r.uncertain ?? ''}</td>
                    <td className="n code" data-label="Rejected">{r.rejected ?? ''}</td>
                    <td className="n code" data-label="Time">{seconds(r.processing_ms)}</td>
                    <td data-label="Analysed"><time dateTime={r.created_at}>{when(r.created_at)}</time></td>
                    <td className="code id" data-label="Id">{r.analysis_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="history__count">
            {items.length} of {total} shown.
            {items.length < total && (
              <button type="button" className="btn btn--outline" onClick={loadMore} disabled={more} aria-busy={more || undefined}>
                {more ? 'Loading…' : 'Show more'}
              </button>
            )}
          </p>
        </>
      )}
    </section>
  );
}
