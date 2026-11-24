import { useCallback, useEffect, useRef, useState } from 'react';
import { deleteCorrection, getCorrections, putCorrection } from './api.js';

export const LABEL_TEXT = { reject: 'Debris (reject)', other_organism: 'Other organism', not_an_object: 'Not an object' };
export const labelText = (l) => LABEL_TEXT[l] ?? l;

// The label corrections of one analysis: load once, save per detection, keep the server's reviewed counts.
// rows[detId] = 'saving' | 'saved' | { error } while or after a save.
// Saves on different rows may overlap. Every answer carries the whole view, so only the answer to the
// LATEST request is applied: an older answer arriving late can never undo a newer correction on screen.
export default function useCorrections(analysisId) {
  const [state, setState] = useState('loading'); // loading | ready | error
  const [labels, setLabels] = useState([]);
  const [byDet, setByDet] = useState({});
  const [counts, setCounts] = useState(null);
  const [rows, setRows] = useState({});
  const [error, setError] = useState(null);
  const seq = useRef(0);

  const apply = useCallback((view, n) => {
    if (n !== seq.current) return;
    setLabels(view.labels);
    setByDet(Object.fromEntries(view.items.map((c) => [c.detection_id, c])));
    setCounts(view.reviewed_counts);
  }, []);

  useEffect(() => {
    let live = true;
    const n = ++seq.current;
    setState('loading');
    getCorrections(analysisId)
      .then((v) => { if (live) { apply(v, n); setState('ready'); } })
      .catch((err) => { if (live) { setError(err); setState('error'); } });
    return () => { live = false; };
  }, [analysisId, apply]);

  const save = useCallback(async (detId, label) => {
    const n = ++seq.current;
    setRows((r) => ({ ...r, [detId]: 'saving' }));
    setByDet((b) => {                                 // show the choice at once; the server's view follows
      const next = { ...b };
      if (label) next[detId] = { ...(b[detId] || { detection_id: detId }), corrected_label: label };
      else delete next[detId];
      return next;
    });
    try {
      const view = label ? await putCorrection(analysisId, detId, label) : await deleteCorrection(analysisId, detId);
      apply(view, n);
      setRows((r) => ({ ...r, [detId]: 'saved' }));
    } catch (err) {
      setRows((r) => ({ ...r, [detId]: { error: err.message } }));
      const m = ++seq.current;                         // re-read the truth after a failed save
      getCorrections(analysisId).then((v) => apply(v, m)).catch(() => {});
    }
  }, [analysisId, apply]);

  return { state, error, labels, byDet, counts, rows, save };
}
