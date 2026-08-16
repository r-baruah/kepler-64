/**
 * Kepler-64 Multiverse Worker (3.1)
 * Replays a game and scores every ply across the multiverse off the main thread.
 */

import { Chess } from 'chess.js';
import { sampleUniverses, evaluateAcrossUniverses } from './multiverse';
import type { ConstantsConfig } from './constants';

const scope = self as unknown as {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage: (msg: unknown) => void;
};

scope.onmessage = (e: MessageEvent) => {
  const { startFen, moves, config, sampleCount } = (e.data ?? {}) as {
    startFen?: string;
    moves?: string[];
    config?: ConstantsConfig;
    sampleCount?: number;
  };
  try {
    if (!config || !Array.isArray(moves)) throw new Error('Bad multiverse request');
    const samples = sampleUniverses(config, sampleCount ?? 5);
    const chess = new Chess(startFen);
    const points = moves.map((san, i) => {
      chess.move(san);
      const { mean, sigma, scores } = evaluateAcrossUniverses(chess.fen(), samples);
      return { ply: i, moveSan: san, mean, sigma, spaghetti: scores };
    });
    scope.postMessage({ ok: true, points });
  } catch (err) {
    scope.postMessage({ ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
