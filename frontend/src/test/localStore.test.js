import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { allRecords, clearRecords, deleteRecord, getRecord, listRecords, putRecord, resetLocalStore } from '../localStore.js';

// The IndexedDB path (jsdom has no IndexedDB; fake-indexeddb implements the standard API in memory).
const rec = (id, at, extra = {}) => ({ analysis_id: id, created_at: at, filename: `${id}.png`, status: 'ok', ...extra });

describe('localStore on IndexedDB', () => {
  it('stores, lists newest first, filters, deletes and clears', async () => {
    resetLocalStore();
    expect(typeof indexedDB).toBe('object');
    await putRecord(rec('a', '2026-11-20T10:00:00.000000+00:00'));
    await putRecord(rec('b', '2026-11-21T10:00:00.000Z', { status: 'failed' }));
    await putRecord(rec('c', '2026-11-20T12:00:00.000000+00:00', { review: { items: [{ detection_id: 1 }] } }));
    expect((await allRecords()).map((r) => r.analysis_id)).toEqual(['b', 'c', 'a']);
    expect(await listRecords({ limit: 1, offset: 1 })).toEqual({ items: [expect.objectContaining({ analysis_id: 'c' })], total: 3 });
    expect((await listRecords({ status: 'failed' })).items.map((r) => r.analysis_id)).toEqual(['b']);
    expect((await getRecord('c')).review.items).toEqual([{ detection_id: 1 }]);
    await putRecord({ ...(await getRecord('c')), review: { items: [] } });       // replace, not duplicate
    expect((await getRecord('c')).review.items).toEqual([]);
    expect((await allRecords())).toHaveLength(3);
    expect(await deleteRecord('a')).toBe(true);
    expect(await deleteRecord('a')).toBe(false);
    expect(await getRecord('a')).toBeNull();
    expect(await clearRecords()).toBe(2);
    expect(await allRecords()).toEqual([]);
  });

  it('keeps the data across a new connection (a page reload)', async () => {
    resetLocalStore();
    await putRecord(rec('x', '2026-11-22T10:00:00Z', { result: { counts: { confirmed_total: 3 } } }));
    resetLocalStore();                         // forget the open connection, as a reload does; the database remains
    expect((await getRecord('x')).result.counts.confirmed_total).toBe(3);
  });
});
