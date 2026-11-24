import { useCallback, useEffect, useState } from 'react';
import { getHealth, getInfo, listExamples, setStorage } from './api.js';
import AnalysisDetail from './components/AnalysisDetail.jsx';
import AnalyzeView from './components/AnalyzeView.jsx';
import HistoryView from './components/HistoryView.jsx';

// View state lives in the URL: ?view=analyze | ?view=history | ?view=history&analysis=<id>
function readLocation() {
  const q = new URLSearchParams(window.location.search);
  const analysis = q.get('analysis');
  return { view: analysis || q.get('view') === 'history' ? 'history' : 'analyze', analysis };
}

export default function App() {
  const [loc, setLoc] = useState(readLocation);
  const [health, setHealth] = useState(null);
  const [info, setInfo] = useState(null);
  const [examples, setExamples] = useState([]);
  const [bootError, setBootError] = useState(null);
  const [historyKey, setHistoryKey] = useState(0);

  const go = useCallback((view, analysis = null) => {
    const q = new URLSearchParams();
    q.set('view', view);
    if (analysis) q.set('analysis', analysis);
    window.history.pushState(null, '', `?${q}`);
    setLoc({ view, analysis });
    window.scrollTo({ top: 0 });
  }, []);

  useEffect(() => {
    const onPop = () => setLoc(readLocation());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    let live = true;
    getHealth()
      .then(async (h) => {
        if (!live) return;
        setStorage(h.storage);
        setHealth(h);
        if (h.models_loaded) {
          const [i, ex] = await Promise.all([getInfo(), listExamples().catch(() => [])]);
          if (live) { setInfo(i); setExamples(ex); }
        }
      })
      .catch((err) => live && setBootError(err));
    return () => { live = false; };
  }, []);

  const ready = Boolean(health?.models_loaded && info);
  const booted = Boolean(health || bootError);   // the history views need to know where the history is
  const tab = (view, label) => (
    <a
      href={`?view=${view}`}
      className="nav__link"
      aria-current={loc.view === view ? 'page' : undefined}
      onClick={(e) => { e.preventDefault(); go(view); }}
    >
      {label}
    </a>
  );

  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <header className="topbar">
        <div className="container topbar__inner">
          <a className="wordmark" href="?view=analyze" aria-label="PlanktoScan AI, analyse an image"
            onClick={(e) => { e.preventDefault(); go('analyze'); }}>
            PlanktoScan
          </a>
          <nav className="nav" aria-label="Main">
            {tab('analyze', 'Analyse')}
            {tab('history', 'History')}
          </nav>
          <p className="engine" role="status">
            {bootError ? 'Server offline'
              : !health ? 'Connecting…'
                : health.models_loaded ? <>Models ready on <span className="code">{health.device}</span></>
                  : 'Models not loaded'}
          </p>
        </div>
      </header>

      <main id="main" className="container main">
        {bootError && (
          <div className="alert banner" role="alert">
            <h2>Server not reachable</h2>
            <p>{bootError.message}</p>
          </div>
        )}
        {health && !health.models_loaded && (
          <div className="alert banner" role="alert">
            <h2>Models not loaded</h2>
            <p>{health.error || 'The server started without the models.'} History stays available.</p>
          </div>
        )}

        {loc.view === 'analyze' && (
          <AnalyzeView
            info={info}
            examples={examples}
            ready={ready}
            onAnalysed={() => setHistoryKey((k) => k + 1)}
            onOpen={(id) => go('history', id)}
          />
        )}
        {loc.view === 'history' && !booted && (
          <div className="skeleton skeleton--table" role="status" aria-label="Connecting to the server">
            {Array.from({ length: 5 }, (_, i) => <span key={i} />)}
          </div>
        )}
        {booted && loc.view === 'history' && !loc.analysis && (
          <HistoryView refreshKey={historyKey} onOpen={(id) => (id ? go('history', id) : go('analyze'))} />
        )}
        {booted && loc.view === 'history' && loc.analysis && (
          <AnalysisDetail
            id={loc.analysis}
            onBack={() => go('history')}
            onDeleted={() => { setHistoryKey((k) => k + 1); go('history'); }}
          />
        )}
      </main>

      <footer className="footer">
        <div className="container footer__inner">
          <p>
            Trained on ZooLake (Eawag, freshwater, doi 10.25678/0004DY). The detector was trained and evaluated on
            synthetic frames: results on real microscope images are not validated.
          </p>
          <a className="link" href="/docs">API reference</a>
        </div>
      </footer>
    </>
  );
}
