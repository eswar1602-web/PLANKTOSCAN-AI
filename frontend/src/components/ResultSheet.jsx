import { useState } from 'react';
import SpecimenFrame from './SpecimenFrame.jsx';
import useCorrections, { labelText } from '../useCorrections.js';
import { DET_STATUS_TEXT, STATUS_TEXT, countRange, countRows, rate, seconds, when } from '../format.js';

const FORMATS = [
  { key: 'pdf', label: 'PDF report' },
  { key: 'html', label: 'HTML report' },
  { key: 'csv', label: 'CSV' },
  { key: 'json', label: 'JSON' },
];

// One analysis as a specification sheet: figures, the specimen frame, counts, detections, downloads.
export default function ResultSheet({ result, links, actions }) {
  const [active, setActive] = useState(null);
  const review = useCorrections(result.analysis_id);
  const rc = review.counts;
  const counts = countRows(result);
  const t = result.timings_ms;
  const th = result.metadata.thresholds;
  const stages = Object.entries(t).filter(([k]) => k !== 'total');
  const hasRange = Boolean(result.count_intervals?.per_class);
  const reject = result.metadata.classes.reject;
  const conc = result.concentration;
  const factor = conc?.calibration?.concentration_factor ?? 1;
  const sens = result.threshold_sensitivity;
  const calibrated = result.detections.some((d) => d.calibrated_prob !== null && d.calibrated_prob !== undefined);
  const unknown = result.counts.possibly_unknown ?? 0;

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
                {hasRange && <th scope="col" className="n">95% range</th>}
              </tr>
            </thead>
            <tbody>
              {counts.rows.map((r) => (
                <tr key={r.key} className={r.confirmed + r.estimated === 0 ? 'is-zero' : undefined}>
                  <th scope="row">{r.label}</th>
                  <td className="n code">{r.confirmed}</td>
                  <td className="n code">{r.estimated}</td>
                  <td className="n code">{r.fromUncertain}</td>
                  {hasRange && <td className="n code">{countRange(result, r.key)}</td>}
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
                  {hasRange && <td className="n code">{r.key === 'rejected' ? countRange(result, reject) : ''}</td>}
                </tr>
              ))}
              <tr>
                <th scope="row">Uncertain</th>
                <td className="n code">{counts.uncertain}</td>
                <td className="n">not counted</td>
                <td className="n" />
                {hasRange && <td className="n" />}
              </tr>
            </tbody>
          </table>
          </div>
          <p className="note">
            Confirmed counts only confident organisms (probability at least {th.uncertainty.toFixed(2)}). Estimated
            puts every detection under its most likely class.
            {hasRange && ' The 95% range comes from the class probabilities of all detections; it does not include missed or false detections.'}
          </p>
          {sens && (
            <p className="note" data-testid="sensitivity">
              {`Threshold sensitivity: ${sens.lower.confirmed_total}, ${sens.chosen.confirmed_total} and `}
              {`${sens.higher.confirmed_total} confirmed organisms at uncertainty ${sens.lower.threshold.toFixed(2)}, `}
              {`${sens.chosen.threshold.toFixed(2)} and ${sens.higher.threshold.toFixed(2)}; `}
              {`${sens.borderline} of ${counts.detections} detections are within ${sens.delta.toFixed(2)} of the threshold.`}
            </p>
          )}
          {unknown > 0 && (
            <p className="note">
              {`${unknown} ${unknown === 1 ? 'detection looks' : 'detections look'} unlike every trained class `}
              {`(marked possibly unknown below; ${result.counts.possibly_unknown_confirmed} in the confirmed counts).`}
            </p>
          )}
        </section>
      </div>

      <section className="review" aria-labelledby="review-title">
        <h3 id="review-title">Your review</h3>
        {review.state === 'error' && <p className="alert" role="alert">Corrections could not be loaded: {review.error?.message}</p>}
        {review.state === 'ready' && rc && rc.reviewed === 0 && (
          <p className="note">
            Check detections against the image and set the right label in the last column of the detection
            table. Corrections are kept with this analysis, give reviewed counts here, and can be exported from
            the history for retraining. The model&apos;s counts above never change.
          </p>
        )}
        {review.state === 'ready' && rc && rc.reviewed > 0 && (
          <>
            <div className="table-scroll" tabIndex={0} aria-label="Reviewed counts table, scrolls sideways">
              <table className="spec spec--counts">
                <thead>
                  <tr>
                    <th scope="col">Class</th>
                    <th scope="col" className="n">Model, confirmed</th>
                    <th scope="col" className="n">After review</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(rc.confirmed).map(([k, v]) => (
                    <tr key={k} className={v + result.counts.confirmed[k] === 0 ? 'is-zero' : undefined}>
                      <th scope="row">{k}</th>
                      <td className="n code">{result.counts.confirmed[k]}</td>
                      <td className="n code">{v}</td>
                    </tr>
                  ))}
                  <tr className={rc.other_organism === 0 ? 'is-zero' : undefined}>
                    <th scope="row">Other organism</th>
                    <td className="n code">-</td>
                    <td className="n code">{rc.other_organism}</td>
                  </tr>
                </tbody>
                <tbody className="spec__totals">
                  <tr>
                    <th scope="row">All organisms</th>
                    <td className="n code">{result.counts.confirmed_total}</td>
                    <td className="n code">{rc.organisms_total}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="note" data-testid="review-summary">
              {`Reviewed ${rc.reviewed} of ${rc.detections} detections: ${rc.agreed} agreed with the model, ${rc.changed} changed. `}
              {rc.unreviewed_uncertain > 0 && `${rc.unreviewed_uncertain} uncertain ${rc.unreviewed_uncertain === 1 ? 'detection is' : 'detections are'} not reviewed yet.`}
            </p>
          </>
        )}
      </section>

      {conc && (
        <section className="concentration" aria-labelledby="conc-title">
          <h3 id="conc-title">Concentration, per litre of original water</h3>
          <div className="table-scroll" tabIndex={0} aria-label="Concentration table, scrolls sideways">
            <table className="spec spec--counts">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col" className="n">Confirmed</th>
                  <th scope="col" className="n">Per litre</th>
                  <th scope="col" className="n">95% interval</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(conc.per_class).map(([k, v]) => (
                  <tr key={k} className={v.count === 0 ? 'is-zero' : undefined}>
                    <th scope="row">{k}</th>
                    <td className="n code">{v.count}</td>
                    <td className="n code">{rate(v.per_l)}</td>
                    <td className="n code">{rate(v.per_l_low)} to {rate(v.per_l_high)}</td>
                  </tr>
                ))}
              </tbody>
              <tbody className="spec__totals">
                <tr>
                  <th scope="row">All organisms</th>
                  <td className="n code">{conc.total.count}</td>
                  <td className="n code">{rate(conc.total.per_l)}</td>
                  <td className="n code">{rate(conc.total.per_l_low)} to {rate(conc.total.per_l_high)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="note">
            {`${rate(conc.imaged_volume_ml)} mL imaged${factor !== 1 ? `, concentrated ${rate(factor)} times` : ''}, `}
            {`so ${rate(conc.water_volume_ml)} mL of original water. `}
            The interval is counting error only (Poisson). Pool several images of one sample with the
            command-line tool for a narrower one.
          </p>
        </section>
      )}

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
                  {calibrated && <th scope="col" className="n">Calibrated</th>}
                  <th scope="col" className="n">Detector confidence</th>
                  <th scope="col" className="n">Box (x0, y0, x1, y1)</th>
                  <th scope="col">Your label</th>
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
                    <td data-label="Status">
                      <span className={`tag tag--${d.status}`}>{DET_STATUS_TEXT[d.status]}</span>
                      {d.unknown && <span className="tag tag--unknown">Possibly unknown</span>}
                    </td>
                    <td className="n code" data-label="Class probability">{d.prob.toFixed(3)}</td>
                    {calibrated && <td className="n code" data-label="Calibrated">{d.calibrated_prob?.toFixed(3) ?? ''}</td>}
                    <td className="n code" data-label="Detector confidence">{d.det_conf.toFixed(3)}</td>
                    <td className="n code" data-label="Box">{[d.box.x0, d.box.y0, d.box.x1, d.box.y1].map((v) => Math.round(v)).join(', ')}</td>
                    <td data-label="Your label" className="review-cell">
                      <select
                        className="input input--compact"
                        aria-label={`Your label for detection ${d.id}`}
                        value={review.byDet[d.id]?.corrected_label ?? ''}
                        disabled={review.state !== 'ready' || review.rows[d.id] === 'saving'}
                        onChange={(e) => review.save(d.id, e.target.value)}
                      >
                        <option value="">Not reviewed</option>
                        {review.labels.map((l) => (
                          <option key={l} value={l}>{l === d.label ? `${labelText(l)} (as model)` : labelText(l)}</option>
                        ))}
                      </select>
                      <span className="review-cell__state" aria-live="polite">
                        {review.rows[d.id] === 'saving' && 'Saving'}
                        {review.rows[d.id] === 'saved' && 'Saved'}
                        {review.rows[d.id]?.error && <span className="row-error">{review.rows[d.id].error}</span>}
                      </span>
                    </td>
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
