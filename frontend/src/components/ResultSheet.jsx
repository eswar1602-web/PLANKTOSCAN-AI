import { useState } from 'react';
import SpecimenFrame from './SpecimenFrame.jsx';
import { DET_STATUS_TEXT, STATUS_TEXT, countRows, seconds, when } from '../format.js';

const FORMATS = [
  { key: 'pdf', label: 'PDF report' },
  { key: 'html', label: 'HTML report' },
  { key: 'csv', label: 'CSV' },
  { key: 'json', label: 'JSON' },
];

// One analysis as a specification sheet: figures, the specimen frame, counts, detections, downloads.
export default function ResultSheet({ result, links, actions }) {
  const [active, setActive] = useState(null);
  const counts = countRows(result);
  const t = result.timings_ms;
  const th = result.metadata.thresholds;
  const stages = Object.entries(t).filter(([k]) => k !== 'total');

  return (
    <article className="sheet" aria-labelledby="sheet-title">
      <header className="sheet__head">
        <div className="sheet__title">
          <h2 id="sheet-title">{result.filename}</h2>
          <p className="sheet__meta">
            <span className={`status status--${result.status}`} role="status">{STATUS_TEXT[result.status]}</span>
            <span className="code">{result.analysis_id}</span>
            <time dateTime={result.created_at}>{when(result.created_at)}</time>
          </p>
        </div>
        {actions}
      </header>

      {result.warnings.length > 0 && (
        <ul className="notices" aria-label="Warnings">
          {result.warnings.map((w) => <li key={w}>{w}</li>)}
        </ul>
      )}

      <dl className="figures">
        <div><dt>Confirmed organisms</dt><dd className="figure">{counts.total.confirmed}</dd></div>
        <div><dt>Estimated organisms</dt><dd className="figure">{counts.total.estimated}</dd></div>
        <div><dt>Uncertain</dt><dd className="figure">{counts.uncertain}</dd></div>
        <div><dt>Rejected</dt><dd className="figure">{counts.rejected.confirmed}</dd></div>
        <div><dt>Processing time</dt><dd className="figure">{seconds(t.total)}</dd></div>
      </dl>

      <div className="sheet__body">
        <SpecimenFrame
          images={Object.fromEntries(Object.entries(links.images))}
          width={result.metadata.image.width}
          height={result.metadata.image.height}
          detections={result.detections}
          active={active}
          onActive={setActive}
          filename={result.filename}
        />

        <section className="counts" aria-labelledby="counts-title">
          <h3 id="counts-title">Counts by class</h3>
          <div className="table-scroll" tabIndex={0} aria-label="Counts table, scrolls sideways">
          <table className="spec spec--counts">
            <thead>
              <tr>
                <th scope="col">Class</th>
                <th scope="col" className="n">Confirmed</th>
                <th scope="col" className="n">Estimated</th>
                <th scope="col" className="n">From uncertain</th>
              </tr>
            </thead>
            <tbody>
              {counts.rows.map((r) => (
                <tr key={r.key} className={r.confirmed + r.estimated === 0 ? 'is-zero' : undefined}>
                  <th scope="row">{r.label}</th>
                  <td className="n code">{r.confirmed}</td>
                  <td className="n code">{r.estimated}</td>
                  <td className="n code">{r.fromUncertain}</td>
                </tr>
              ))}
            </tbody>
            <tbody className="spec__totals">
              {[counts.total, counts.rejected].map((r) => (
                <tr key={r.key}>
                  <th scope="row">{r.label}</th>
                  <td className="n code">{r.confirmed}</td>
                  <td className="n code">{r.estimated}</td>
                  <td className="n code">{r.fromUncertain}</td>
                </tr>
              ))}
              <tr>
                <th scope="row">Uncertain</th>
                <td className="n code">{counts.uncertain}</td>
                <td className="n">not counted</td>
                <td className="n" />
              </tr>
            </tbody>
          </table>
          </div>
          <p className="note">
            Confirmed counts only confident organisms (probability at least {th.uncertainty.toFixed(2)}). Estimated
            puts every detection under its most likely class.
          </p>
        </section>
      </div>

      <section className="detections" aria-labelledby="det-title">
        <h3 id="det-title">Detections, {counts.detections}</h3>
        {result.detections.length === 0 ? (
          <p className="empty">{result.message} The detector found nothing above confidence {th.det_conf.toFixed(2)}.</p>
        ) : (
          <div className="table-scroll" tabIndex={0} aria-label="Detection table, scrolls sideways">
            <table className="spec spec--rows">
              <thead>
                <tr>
                  <th scope="col">Id</th>
                  <th scope="col">Class</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="n">Class probability</th>
                  <th scope="col" className="n">Detector confidence</th>
                  <th scope="col" className="n">Box (x0, y0, x1, y1)</th>
                </tr>
              </thead>
              <tbody>
                {result.detections.map((d) => (
                  <tr
                    key={d.id}
                    tabIndex={0}
                    className={active === d.id ? 'is-active' : undefined}
                    onPointerEnter={() => setActive(d.id)}
                    onPointerLeave={() => setActive(null)}
                    onFocus={() => setActive(d.id)}
                    onBlur={() => setActive(null)}
                  >
                    <td className="code id" data-label="Id">{d.id}</td>
                    <td data-label="Class">{d.label}</td>
                    <td data-label="Status"><span className={`tag tag--${d.status}`}>{DET_STATUS_TEXT[d.status]}</span></td>
                    <td className="n code" data-label="Class probability">{d.prob.toFixed(3)}</td>
                    <td className="n code" data-label="Detector confidence">{d.det_conf.toFixed(3)}</td>
                    <td className="n code" data-label="Box">{[d.box.x0, d.box.y0, d.box.x1, d.box.y1].map((v) => Math.round(v)).join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="sheet__foot">
        <div className="downloads" aria-label="Download reports">
          {FORMATS.filter((f) => links.reports[f.key]).map((f) => (
            <a key={f.key} className="btn btn--outline" href={links.reports[f.key]} download>
              {f.label}
            </a>
          ))}
        </div>
        <dl className="timings" aria-label="Processing time by stage">
          {stages.map(([k, v]) => (
            <div key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd className="code">{Math.round(v)}&nbsp;ms</dd></div>
          ))}
          <div><dt>Thresholds</dt><dd className="code">det {th.det_conf.toFixed(2)}, unc {th.uncertainty.toFixed(2)}</dd></div>
        </dl>
      </section>
    </article>
  );
}
