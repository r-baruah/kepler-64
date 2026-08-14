/**
 * Kepler-64 Score Decomposition and Full Position Evaluation
 */

import type { ConstantsConfig } from './constants';
import { DEFAULT_CONSTANTS } from './constants';
import { KeplerBoard } from './board';
import { potentialField, forceField, DIST_64, DIST2_64 } from './gravity';
import { computeTidalDisruption } from './tidal';
import type { TidalDisruptionResult } from './tidal';

export interface ScoreBreakdown {
  enemyKingTide: number;     // +eta_b (good for mover)
  ownKingTide: number;       // -eta_w (bad for mover)
  forceEnemyKing: number;    // +bonus_b
  forceOwnKing: number;      // +pen_w (negative)
  bindingEnergy: number;     // Army coordination edge
  materialBalance: number;   // Gravitational mass edge
  totalScoreWhite: number;
  totalScoreMover: number;
  whiteKingTidal: TidalDisruptionResult | null;
  blackKingTidal: TidalDisruptionResult | null;
}

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

export function evaluatePosition(
  board: KeplerBoard,
  config: ConstantsConfig = DEFAULT_CONSTANTS
): ScoreBreakdown {
  const masses = board.massVector();
  const whiteM = new Float32Array(64);
  const blackM = new Float32Array(64);

  let sumWhiteM = 0;
  let sumBlackM = 0;

  for (let i = 0; i < 64; i++) {
    const m = masses[i];
    if (m > 0) {
      whiteM[i] = m;
      sumWhiteM += m;
    } else if (m < 0) {
      blackM[i] = -m;
      sumBlackM += -m;
    }
  }

  // Source-attributed fields
  const Uw = potentialField(whiteM, config.eps, config.G, config.c);
  const Ub = potentialField(blackM, config.eps, config.G, config.c);
  const Fw = forceField(whiteM, config.eps, config.G, config.c);
  const Fb = forceField(blackM, config.eps, config.G, config.c);

  const wk = board.findKingSquare('w');
  const bk = board.findKingSquare('b');

  let etaB = 0;
  let etaW = 0;
  let bonusB = 0;
  let penW = 0;

  let whiteKingTidal: TidalDisruptionResult | null = null;
  let blackKingTidal: TidalDisruptionResult | null = null;

  if (bk !== null) {
    blackKingTidal = computeTidalDisruption(Uw, bk, config.roche, config.Rg, config.mref);
    etaB = blackKingTidal.eta;
    const fMagBk = Math.sqrt(Fw.fx[bk] * Fw.fx[bk] + Fw.fy[bk] * Fw.fy[bk]);
    bonusB = config.bonus * sigmoid(config.kgain * (fMagBk - config.roche));
  }

  if (wk !== null) {
    whiteKingTidal = computeTidalDisruption(Ub, wk, config.roche, config.Rg, config.mref);
    etaW = whiteKingTidal.eta;
    const fMagWk = Math.sqrt(Fb.fx[wk] * Fb.fx[wk] + Fb.fy[wk] * Fb.fy[wk]);
    penW = -config.bonus * sigmoid(config.kgain * (fMagWk - config.roche));
  }

  // Binding Energy Edge (Excluding kings as detectors)
  let bindingEdge = 0;
  if (wk !== null && bk !== null) {
    const whiteCo = new Float32Array(whiteM);
    const blackCo = new Float32Array(blackM);
    whiteCo[wk] = 0;
    blackCo[bk] = 0;

    let dotW = 0;
    let dotB = 0;
    let selfDotW = 0;
    let selfDotB = 0;

    const selfScale = (config.G * sigmoid(config.c)) / Math.sqrt(config.eps * config.eps);

    for (let i = 0; i < 64; i++) {
      // White army binding
      const dWk = DIST_64[i * 64 + wk];
      const rWk = Math.sqrt(DIST2_64[i * 64 + wk] + config.eps * config.eps);
      const uWc = Uw[i] + (1000.0 * sigmoid(config.c - dWk)) / rWk;
      dotW += whiteCo[i] * uWc;
      selfDotW += whiteCo[i] * whiteCo[i];

      // Black army binding
      const dBk = DIST_64[i * 64 + bk];
      const rBk = Math.sqrt(DIST2_64[i * 64 + bk] + config.eps * config.eps);
      const uBc = Ub[i] + (1000.0 * sigmoid(config.c - dBk)) / rBk;
      dotB += blackCo[i] * uBc;
      selfDotB += blackCo[i] * blackCo[i];
    }

    const bindW = (dotW + selfScale * selfDotW) / 2.0;
    const bindB = (dotB + selfScale * selfDotB) / 2.0;
    bindingEdge = config.gamma * (bindB - bindW);
  }

  // Material Advantage
  const material = config.mat_gain * ((sumWhiteM - 1000.0) - (sumBlackM - 1000.0));

  const totalScoreWhite = etaB - etaW + bonusB + penW + bindingEdge + material;
  const totalScoreMover = board.turn === 'w' ? totalScoreWhite : -totalScoreWhite;

  return {
    enemyKingTide: board.turn === 'w' ? etaB : etaW,
    ownKingTide: board.turn === 'w' ? -etaW : -etaB,
    forceEnemyKing: board.turn === 'w' ? bonusB : -penW,
    forceOwnKing: board.turn === 'w' ? penW : -bonusB,
    bindingEnergy: board.turn === 'w' ? bindingEdge : -bindingEdge,
    materialBalance: board.turn === 'w' ? material : -material,
    totalScoreWhite,
    totalScoreMover,
    whiteKingTidal,
    blackKingTidal,
  };
}
