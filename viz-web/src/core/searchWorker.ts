/**
 * Kepler-64 Search Worker (2.1)
 * Runs gravitational beam search off the main thread.
 */

import { searchBestMove } from './search';
import { getPersona } from './personas';

const scope = self as unknown as {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage: (msg: unknown) => void;
};

scope.onmessage = (e: MessageEvent) => {
  const { fen, personaId } = (e.data ?? {}) as { fen?: string; personaId?: string };
  try {
    if (!fen || !personaId) throw new Error('Bad search request');
    const persona = getPersona(personaId);
    const result = searchBestMove(fen, persona);
    scope.postMessage({ ok: true, result });
  } catch (err) {
    scope.postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
