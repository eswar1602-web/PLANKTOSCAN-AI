import { useEffect, useRef, useState } from 'react';
import { analyze } from '../api.js';
import { calibrationError, validThreshold, validateFile } from '../format.js';
import ResultSheet from './ResultSheet.jsx';

const SLOW_MS = 2500;
const CAL_FIELDS = [
  ['volumeMl', 'cal-vol', 'Imaged volume, mL', 'e.g. 0.002'],
  ['factor', 'cal-factor', 'Concentration factor', '1'],
  ['pixelUm', 'cal-px', 'Pixel size, µm', 'e.g. 1.2'],
  ['depthUm', 'cal-depth', 'Field depth, µm', 'e.g. 500'],
];

export default function AnalyzeView({ info, examples, ready, onAnalysed, onOpen }) {
  const [file, setFile] = useState(null);
  const [example, setExample] = useState('');
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [detConf, setDetConf] = useState('');
  const [unc, setUnc] = useState('');
  const [cal, setCal] = useState({ volumeMl: '', pixelUm: '', depthUm: '', factor: '' });
  const [fieldError, setFieldError] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle | busy | error | done
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState(null);
  const [out, setOut] = useState(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (!file) { setPreview(null); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const defaults = info?.defaults;
  const chosen = file ? file.name : example;

  function choose(f) {
    setExample('');
    setOut(null);
    setError(null);
    const problem = validateFile(f, info);
    setFieldError(problem);
    setFile(problem ? null : f);
  }

  async function submit(e) {
    e.preventDefault();
    if (!file && !example) { setFieldError('Choose an image or an example first.'); inputRef.current?.focus(); return; }
    if (!validThreshold(detConf) || !validThreshold(unc)) { setFieldError('Thresholds are numbers from 0 to 1.'); return; }
    const calProblem = calibrationError(cal);
    if (calProblem) { setFieldError(calProblem); return; }
    setFieldError(null);
    setError(null);
    setPhase('busy');
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), SLOW_MS);
    abortRef.current = new AbortController();
    try {
      const body = await analyze({ file, example: file ? null : example, detConf, uncertainty: unc, calibration: cal }, abortRef.current.signal);
      setOut(body);
      setPhase('done');
      onAnalysed?.(body.result.analysis_id);
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err);
      setPhase('error');
    } finally {
      clearTimeout(timer);
    }
  }

  const busy = phase === 'busy';
  const blockedWhy = !ready ? 'The models are not loaded, so nothing can be analysed yet.' : null;

  return (
    <div className="workbench">
      <form className="intake card" onSubmit={submit} noValidate aria-describedby="intake-help">
        <h2>Analyse an image</h2>
        <label
          className={`drop${dragging ? ' is-over' : ''}${fieldError ? ' is-invalid' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files[0]) choose(e.dataTransfer.files[0]); }}
        >
          <input
            ref={inputRef}
            type="file"
            name="file"
            accept={(info?.formats ?? []).join(',')}
            className="visually-hidden"
            aria-invalid={fieldError ? 'true' : undefined}
            aria-describedby="intake-help"
            onChange={(e) => e.target.files[0] && choose(e.target.files[0])}
          />
          {preview ? (
            <img className="drop__preview" src={preview} alt="" onError={() => setPreview(null)} />
          ) : null}
          <span className="drop__text">
            {chosen ? <><b>{chosen}</b>{file && <span className="code"> {(file.size / 2 ** 20).toFixed(2)}&nbsp;MB</span>}</> : <b>Drop a microscopy image here, or choose a file</b>}
            <span className="drop__hint">
              {(info?.formats ?? []).join(' ')} up to {info?.limits?.max_file_mb ?? 50}&nbsp;MB
            </span>
          </span>
        </label>

        {examples.length > 0 && (
          <div className="field">
            <label htmlFor="example">Or an example synthetic frame</label>
            <select
              id="example"
              className="input"
              value={example}
              onChange={(e) => { setExample(e.target.value); setFile(null); setFieldError(null); setOut(null); setError(null); }}
            >
              <option value="">None</option>
              {examples.map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}
            </select>
          </div>
        )}

        <fieldset className="thresholds">
          <legend>Thresholds</legend>
          <div className="field">
            <label htmlFor="det">Detection confidence</label>
            <input id="det" className="input code" type="number" inputMode="decimal" min="0" max="1" step="0.01"
              placeholder={defaults ? defaults.det_conf.toFixed(2) : ''} value={detConf}
              onChange={(e) => setDetConf(e.target.value)} aria-describedby="det-help" />
            <p className="help" id="det-help">Default {defaults?.det_conf.toFixed(2)}, chosen on validation frames.</p>
          </div>
          <div className="field">
            <label htmlFor="unc">Uncertainty</label>
            <input id="unc" className="input code" type="number" inputMode="decimal" min="0" max="1" step="0.01"
              placeholder={defaults ? defaults.uncertainty.toFixed(2) : ''} value={unc}
              onChange={(e) => setUnc(e.target.value)} aria-describedby="unc-help" />
            <p className="help" id="unc-help">Default {defaults?.uncertainty.toFixed(2)}. Below it a detection is uncertain.</p>
          </div>
        </fieldset>

        <details className="calibration">
          <summary>Sample calibration, for concentrations</summary>
          <p className="help">
            Optional. Give the volume of sample one image shows, or the pixel size and the depth of the
            field. The result then reports organisms per litre with a 95% interval.
          </p>
          <fieldset className="thresholds">
            <legend className="visually-hidden">Sample calibration</legend>
            {CAL_FIELDS.map(([k, id, label, ph]) => (
              <div className="field" key={k}>
                <label htmlFor={id}>{label}</label>
                <input id={id} className="input code" type="number" inputMode="decimal" min="0" step="any"
                  placeholder={ph} value={cal[k]} onChange={(e) => setCal((c) => ({ ...c, [k]: e.target.value }))} />
              </div>
            ))}
          </fieldset>
          <p className="help">Concentration factor: how many times the sample was concentrated before imaging (1 L to 10 mL is 100).</p>
        </details>

        <p className="help help--form" id="intake-help" aria-live="polite">
          {fieldError || blockedWhy || 'Leave a threshold empty to use its default.'}
        </p>

        <button className="btn btn--primary" type="submit" disabled={busy || !ready} aria-busy={busy ? 'true' : undefined}>
          {busy ? 'Analysing…' : 'Analyse'}
        </button>
      </form>

      <section className="outcome" aria-live="polite" aria-busy={busy}>
        {phase === 'idle' && (
          <div className="empty-state">
            <h2>Nothing analysed yet</h2>
            <p>
              Choose a dark-field plankton image. Each object is detected, named as one of{' '}
              {info?.classes?.organisms?.length ?? 7} plankton groups or rejected, and counted.
            </p>
          </div>
        )}
        {busy && (
          <div className="skeleton" role="status">
            <p className="skeleton__label">
              Analysing {chosen}: detection, then classification of every object.
              {slow ? ' The first image after a restart takes longer.' : ''}
            </p>
            <div className="skeleton__frame" />
            <div className="skeleton__rows"><span /><span /><span /></div>
          </div>
        )}
        {phase === 'error' && error && (
          <div className="alert" role="alert">
            <h2>{error.code === 'network_error' ? 'Server not reachable' : 'This image was not analysed'}</h2>
            <p>{error.message}</p>
            <p className="alert__code code">{error.code}{error.status ? `, HTTP ${error.status}` : ''}</p>
          </div>
        )}
        {phase === 'done' && out && (
          <ResultSheet
            result={out.result}
            links={out.links}
            actions={
              <a className="link" href={`?view=history&analysis=${out.result.analysis_id}`}
                onClick={(e) => { e.preventDefault(); onOpen(out.result.analysis_id); }}>
                Open in history
              </a>
            }
          />
        )}
      </section>
    </div>
  );
}
