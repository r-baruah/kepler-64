/**
 * Kepler-64 Accretion Ledger
 * Replays a move list to track mass accreted onto capturing pieces.
 */

import { PIECE_MASSES } from './constants';

export const ACCRETION_ETA = 0.80;

export interface AccretionEvent {
  ply: number;
  from: string;
  to: string;
  captorSquare: number;
  victimType: string;
  victimMass: number;
  gainedMass: number;
}

export interface AccretionLedger {
  excessBySquare: Record<number, number>;
  history: AccretionEvent[];
}

function squareFromAlgebraic(algebraic: string): number {
  const file = algebraic.charCodeAt(0) - 97;
  const rank = parseInt(algebraic.charAt(1), 10);
  return (rank - 1) * 8 + file;
}

export function buildAccretionLedger(
  moves: { from: string; to: string; captured?: string; san: string }[]
): AccretionLedger {
  const excess = new Float32Array(64);
  const history: AccretionEvent[] = [];

  moves.forEach((move, index) => {
    const fromSq = squareFromAlgebraic(move.from);
    const toSq = squareFromAlgebraic(move.to);

    const carried = excess[fromSq];
    excess[fromSq] = 0;

    if (move.captured) {
      const victimMass = PIECE_MASSES[move.captured] ?? 0;
      const gained = ACCRETION_ETA * victimMass;
      excess[toSq] = carried + gained;

      history.push({
        ply: index + 1,
        from: move.from,
        to: move.to,
        captorSquare: toSq,
        victimType: move.captured,
        victimMass,
        gainedMass: gained,
      });
    } else {
      excess[toSq] = carried;
    }
  });

  const excessBySquare: Record<number, number> = {};
  for (let sq = 0; sq < 64; sq++) {
    if (excess[sq] > 1e-6) {
      excessBySquare[sq] = excess[sq];
    }
  }

  return { excessBySquare, history };
}

export function haloRadius(excessMass: number): number {
  return Math.min(0.5, Math.sqrt(Math.max(0, excessMass)) * 0.16);
}
