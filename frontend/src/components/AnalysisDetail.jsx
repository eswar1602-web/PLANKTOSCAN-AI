import { useEffect, useState } from 'react';
import { browserStorage, deleteAnalysis, getAnalysis } from '../api.js';
import { STATUS_TEXT, when } from '../format.js';
import ResultSheet from './ResultSheet.jsx';

// A previous analysis, loaded from the history (GET /api/analyses/{id}).
export default function AnalysisDetail({ id, onBack, onDeleted }) {
  const [data, setData] = useState(null);
  const [state, setState] = useState('loading');
  const [error, setError] = useState(null);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    setState('loading');
    setConfirming(false);
    getAnalysis(id)
      .then((d) => { if (live) { setData(d); setState('ready'); } })
      .catch((err) => { if (live) { setError(err); setState('error'); } });
    return () => { live = false; };
  }, [id]);

  async function remove() {
    setBusy(true);
    try {
      await deleteAnalysis(id);
      onDeleted();
    } catch (err) {
      setError(err);
      setState('error');
    } finally {
      setBusy(false);
    }
  }

  const back = (
    <a className="link back" href="?view=history" onClick={(e) => { e.preventDefault(); onBack(); }}>
      All analyses
    </a>
  );

  const actions = (
    <div className="sheet__actions">
      {confirming ? (
        <>
          <span className="help">
            Delete this analysis, its files and any label corrections on it?
            {browserStorage() ? '' : ' Export corrections from the history first to keep them.'}
          </span>
          <button type="button" className="btn btn--danger" onClick={remove} disabled={busy} aria-busy={busy || undefined}>
            {busy ? 'Deleting…' : 'Delete'}
          </button>
          <button type="button" className="btn btn--outline" onClick={() => setConfirming(false)}>Keep it</button>
        </>
      ) : (
        <button type="button" className="btn btn--outline" onClick={() => setConfirming(true)}>Delete analysis</button>
      )}
    </div>
  );

  return (
    <section className="detail" aria-label="Previous analysis">
      {back}
      {state === 'loading' && (
        <div className="skeleton" role="status" aria-label="Loading the analysis">
          <div className="skeleton__frame" />
          <div className="skeleton__rows"><span /><span /><span /></div>
        </div>
      )}
      {state === 'error' && (
        <div className="alert" role="alert">
          <h2>{error?.code === 'not_found' ? 'This analysis no longer exists' : 'The analysis could not be loaded'}</h2>
          <p>{error?.message}</p>
        </div>
      )}
      {state === 'ready' && data.status === 'failed' && (
        <article className="sheet">
          <header className="sheet__head">
            <div className="sheet__title">
              <h2>{data.filename}</h2>
              <p className="sheet__meta">
                <span className="status status--failed" role="status">{STATUS_TEXT.failed}</span>
                <span className="code">{data.analysis_id}</span>
                <time dateTime={data.created_at}>{when(data.created_at)}</time>
              </p>
            </div>
            {actions}
          </header>
          <div className="alert" role="alert">
            <h3>Not analysed</h3>
            <p>{data.error_message}</p>
            <p className="alert__code code">{data.error_code}</p>
          </div>
        </article>
      )}
      {state === 'ready' && data.status !== 'failed' && data.result && (
        <ResultSheet result={data.result} links={data.links} actions={actions} />
      )}
      {state === 'ready' && data.files_missing && (
        <div className="alert" role="alert">
          <h2>{data.filename}</h2>
          <p>The stored files for this analysis are missing from the server, so only its summary is left.</p>
          {actions}
        </div>
      )}
    </section>
  );
}
