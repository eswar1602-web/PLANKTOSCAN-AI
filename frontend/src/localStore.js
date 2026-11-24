// The analysis history kept in this browser, for a server in browser-storage mode (stateless hosting, see
// ../../api/main.py). Same shapes as the server's history endpoints, so the views do not know the difference.
// IndexedDB holds the records (images as Blobs); without IndexedDB (old browsers, tests) it lives in memory.

const DB_NAME = 'planktoscan';
const STORE = 'analyses';

let memory = new Map();
let dbPromise = null;

function openDb() {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      let req;
      try {
        req = indexedDB.open(DB_NAME, 1);
      } catch {
        resolve(null);
        return;
      }
      req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'analysis_id' });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);                // private mode or blocked: fall back to memory
      req.onblocked = () => resolve(null);
    });
  }
  return dbPromise;
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let out;
    const req = fn(store);
    if (req) req.onsuccess = () => { out = req.result; };
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error('The browser storage refused the write (is it full?).'));
  });
}

async function all() {
  const db = await openDb();
  const rows = db ? await tx(db, 'readonly', (s) => s.getAll()) : [...memory.values()];
  return rows.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

export async function getRecord(id) {
  const db = await openDb();
  return (db ? await tx(db, 'readonly', (s) => s.get(id)) : memory.get(id)) ?? null;
}

export async function putRecord(rec) {
  const db = await openDb();
  if (db) await tx(db, 'readwrite', (s) => s.put(rec));
  else memory.set(rec.analysis_id, rec);
  return rec;
}

export async function deleteRecord(id) {
  const db = await openDb();
  const had = Boolean(await getRecord(id));
  if (db) await tx(db, 'readwrite', (s) => s.delete(id));
  else memory.delete(id);
  return had;
}

export async function clearRecords() {
  const n = (await all()).length;
  const db = await openDb();
  if (db) await tx(db, 'readwrite', (s) => s.clear());
  else memory.clear();
  return n;
}

export async function listRecords({ limit = 50, offset = 0, status } = {}) {
  const rows = await all();
  const picked = status ? rows.filter((r) => r.status === status) : rows;
  return { items: picked.slice(offset, offset + limit), total: rows.length };
}

export const allRecords = all;

// Tests only: forget everything (memory fallback).
export function resetLocalStore() {
  memory = new Map();
  dbPromise = null;
}
