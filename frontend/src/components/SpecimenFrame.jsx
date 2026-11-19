import { useState } from 'react';

const VIEWS = [
  { key: 'original', label: 'Original' },
  { key: 'enhanced', label: 'Enhanced' },
  { key: 'annotated', label: 'Annotated' },
];

const CAPTIONS = {
  original: 'Boxes drawn live from the detection list: solid confirmed, dashed uncertain, dotted rejected.',
  enhanced: 'Illumination correction, denoising and CLAHE, for viewing only. Detection runs on the original.',
  annotated: 'As in the reports: blue confirmed, orange uncertain, grey rejected.',
};

// The analysed frame on a plate, with the detection boxes as an SVG overlay in image pixel coordinates,
// so they line up at any display size. `active` / `onActive` link a box to its row in the detection table.
export default function SpecimenFrame({ images, width, height, detections, active, onActive, filename }) {
  const [view, setView] = useState('original');
  const [failed, setFailed] = useState({});
  const src = images[view];
  const overlay = view !== 'annotated';
  const stroke = Math.max(2, Math.round(Math.min(width, height) / 320));

  return (
    <figure className="frame card">
      <div className="frame__bar">
        <div className="seg" role="tablist" aria-label="Image view">
          {VIEWS.map((v) => (
            <button
              key={v.key}
              type="button"
              role="tab"
              aria-selected={view === v.key}
              className="seg__btn"
              disabled={!images[v.key]}
              onClick={() => setView(v.key)}
            >
              {v.label}
            </button>
          ))}
        </div>
        <span className="frame__size code">{width} × {height}&nbsp;px</span>
      </div>
      <div className="frame__well">
      <div className="frame__stage" style={{ aspectRatio: `${width} / ${height}`, '--ar': width / height }}>
        {src && !failed[view] ? (
          <img
            src={src}
            width={width}
            height={height}
            alt={`${filename}, ${view} view`}
            onError={() => setFailed((f) => ({ ...f, [view]: true }))}
          />
        ) : (
          <p className="frame__missing">This image is not stored for the analysis.</p>
        )}
        {overlay && detections.length > 0 && (
          <svg
            className="frame__boxes"
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="none"
            role="img"
            aria-label={`${detections.length} detection boxes`}
          >
            {detections.map((d) => {
              const { x0, y0, x1, y1 } = d.box;
              const on = active === d.id;
              return (
                <g
                  key={d.id}
                  className={`box box--${d.status}${on ? ' is-active' : ''}`}
                  onPointerEnter={() => onActive(d.id)}
                  onPointerLeave={() => onActive(null)}
                >
                  <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} strokeWidth={on ? stroke * 1.6 : stroke} />
                  <text x={x0} y={Math.max(y0 - stroke * 2, 12 * stroke)} fontSize={11 * stroke}>
                    {d.id}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      </div>
      <figcaption className="frame__caption">{CAPTIONS[view]}</figcaption>
    </figure>
  );
}
