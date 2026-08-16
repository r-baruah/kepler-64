/**
 * Kepler-64 Gravitational Search
 * Lightweight 2-ply beam minimax over the physical evaluator.
 * Runs off-thread inside searchWorker.ts so the UI stays at 60 FPS.
 */

import { Chess } from 'chess.js';
import { KeplerBoard } from './board';
import { evaluatePosition } from './evaluate';
import { DIST_64 } from './gravity';
import type { ConstantsConfig } from './constants';
import type { BotPersona } from './personas';
import { sampleUniverses, evaluateAcrossUniverses } from './multiverse';

export interface SearchResult {
  san: string;
  from: string;
  to: string;
  promotion?: string;
  score: number;   // White perspective after the opponent's best reply
  depth: number;
  pv: string[];
  note: string;
}

interface MoveLike {
  from: string;
  to: string;
}

function squareFromAlgebraic(alg: string): number {
  return (alg.charCodeAt(1) - 49) * 8 + (alg.charCodeAt(0) - 97);
}

function lorentzGamma(from: string, to: string, config: ConstantsConfig): number {
  const fromSq = squareFromAlgebraic(from);
  const toSq = squareFromAlgebraic(to);
  const dist = DIST_64[fromSq * 64 + toSq];
  const ratio = Math.min(0.95, dist / Math.max(0.1, config.c));
  return 1 / Math.sqrt(Math.max(0.05, 1 - ratio * ratio));
}

function buildBoost(
  board: KeplerBoard,
  config: ConstantsConfig,
  baseExcess: Record<number, number> | null,
  move: MoveLike | null
): Float32Array {
  const boost = new Float32Array(64);

  const fromSq = move ? squareFromAlgebraic(move.from) : -1;
  const carried = (baseExcess && fromSq >= 0) ? (baseExcess[fromSq] ?? 0) : 0;

  if (baseExcess) {
    for (const key of Object.keys(baseExcess)) {
      const sq = parseInt(key, 10);
      boost[sq] += baseExcess[sq];
    }
  }

  if (move) {
    const toSq = squareFromAlgebraic(move.to);

    // Relocate the moving piece's accreted mass to its destination.
    if (fromSq >= 0) boost[fromSq] = 0;
    boost[toSq] += carried;

    const gamma = lorentzGamma(move.from, move.to, config);
    const base = board.squares[toSq]?.mass ?? 0;
    const total = base + (baseExcess?.[toSq] ?? 0) + carried;
    boost[toSq] += (gamma - 1) * total;
  }

  return boost;
}

function relocateExcess(
  baseExcess: Record<number, number> | null,
  move: MoveLike | null
): Record<number, number> | null {
  if (!baseExcess) return null;
  if (!move) return { ...baseExcess };

  const fromSq = squareFromAlgebraic(move.from);
  const toSq = squareFromAlgebraic(move.to);
  const relocated: Record<number, number> = { ...baseExcess };
  const carried = relocated[fromSq] ?? 0;
  relocated[fromSq] = 0;
  relocated[toSq] = (relocated[toSq] ?? 0) + carried;
  return relocated;
}

function scoreFenWhite(
  fen: string,
  config: ConstantsConfig,
  baseExcess: Record<number, number> | null,
  move: MoveLike | null
): number {
  const board = new KeplerBoard(fen);
  board.massBoost = buildBoost(board, config, baseExcess, move);
  return evaluatePosition(board, config).totalScoreWhite;
}

function leafScore(
  fen: string,
  persona: BotPersona,
  baseExcess: Record<number, number> | null,
  move: MoveLike | null
): number {
  if (persona.search === 'quantum' && persona.quantumSamples > 0) {
    const board = new KeplerBoard(fen);
    // Lorentz boost uses the persona's base `c` here (a deliberate approximation).
    const boost = buildBoost(board, persona.config, baseExcess, move);
    const samples = sampleUniverses(persona.config, persona.quantumSamples);
    const { mean, sigma } = evaluateAcrossUniverses(fen, samples, boost);
    return mean - persona.riskLambda * sigma;
  }
  return scoreFenWhite(fen, persona.config, baseExcess, move);
}

export function searchBestMove(
  fen: string,
  persona: BotPersona,
  baseExcess: Record<number, number> | null = null
): SearchResult | null {
  const chess = new Chess(fen);
  const color = chess.turn();
  const legal = chess.moves({ verbose: true });
  if (!legal.length) return null;

  const sign = color === 'w' ? 1 : -1;

  // 1-ply: score every legal move from the bot's perspective.
  const candidates = legal.map((m) => {
    chess.move(m);
    const s = leafScore(chess.fen(), persona, baseExcess, m);
    chess.undo();
    return { m, s };
  });
  candidates.sort((a, b) => (b.s * sign) - (a.s * sign));
  const beam = candidates.slice(0, persona.beam);

  // 2-ply: probe the opponent's best reply for each beam candidate.
  const refined = beam.map((cand) => {
    const relocatedExcess = relocateExcess(baseExcess, cand.m);
    chess.move(cand.m);
    let reply: number;
    let replySan = '';
    const replies = chess.moves({ verbose: true });

    if (!replies.length) {
      reply = chess.isCheckmate() ? (sign > 0 ? 10000 : -10000) : 0;
    } else {
      let best: number | null = null;
      for (const r of replies) {
        chess.move(r);
        const s = scoreFenWhite(chess.fen(), persona.config, relocatedExcess, r);
        chess.undo();
        if (best === null || (s * -sign) > (best * -sign)) {
          best = s;
          replySan = r.san;
        }
      }
      reply = best ?? 0;
    }

    chess.undo();
    return { ...cand, reply, replySan };
  });

  refined.sort((a, b) => (b.reply * sign) - (a.reply * sign));
  const best = refined[0];

  const modeLabel = persona.search === 'quantum' ? 'Bayesian expectation' : '2-ply beam';
  return {
    san: best.m.san,
    from: best.m.from,
    to: best.m.to,
    promotion: best.m.promotion,
    score: best.reply,
    depth: 2,
    pv: best.replySan ? [best.m.san, best.replySan] : [best.m.san],
    note: `${persona.label} · ${modeLabel} · ${best.reply >= 0 ? '+' : ''}${best.reply.toFixed(2)} white`,
  };
}
