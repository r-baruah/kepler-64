/**
 * Kepler-64 Gravitational Search
 * Lightweight 2-ply beam minimax over the physical evaluator.
 * Runs off-thread inside searchWorker.ts so the UI stays at 60 FPS.
 */

import { Chess } from 'chess.js';
import { KeplerBoard } from './board';
import { evaluatePosition } from './evaluate';
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
  note: string;
}

function scoreFenWhite(fen: string, config: ConstantsConfig): number {
  const board = new KeplerBoard(fen);
  return evaluatePosition(board, config).totalScoreWhite;
}

function leafScore(fen: string, persona: BotPersona): number {
  if (persona.search === 'quantum' && persona.quantumSamples > 0) {
    const samples = sampleUniverses(persona.config, persona.quantumSamples);
    const { mean, sigma } = evaluateAcrossUniverses(fen, samples);
    return mean - persona.riskLambda * sigma;
  }
  return scoreFenWhite(fen, persona.config);
}

export function searchBestMove(fen: string, persona: BotPersona): SearchResult | null {
  const chess = new Chess(fen);
  const color = chess.turn();
  const legal = chess.moves({ verbose: true });
  if (!legal.length) return null;

  const sign = color === 'w' ? 1 : -1;

  // 1-ply: score every legal move from the bot's perspective.
  const candidates = legal.map((m) => {
    chess.move(m);
    const s = leafScore(chess.fen(), persona);
    chess.undo();
    return { m, s };
  });
  candidates.sort((a, b) => (b.s * sign) - (a.s * sign));
  const beam = candidates.slice(0, persona.beam);

  // 2-ply: probe the opponent's best reply for each beam candidate.
  const refined = beam.map((cand) => {
    chess.move(cand.m);
    let reply: number;
    const replies = chess.moves({ verbose: true });

    if (!replies.length) {
      reply = chess.isCheckmate() ? (sign > 0 ? 10000 : -10000) : 0;
    } else {
      let best: number | null = null;
      for (const r of replies) {
        chess.move(r);
        const s = scoreFenWhite(chess.fen(), persona.config);
        chess.undo();
        if (best === null || (s * -sign) > (best * -sign)) best = s;
      }
      reply = best ?? 0;
    }

    chess.undo();
    return { ...cand, reply };
  });

  refined.sort((a, b) => (b.reply * sign) - (a.reply * sign));
  const best = refined[0];

  const modeLabel = persona.search === 'quantum' ? 'Bayesian risk' : '2-ply beam';
  return {
    san: best.m.san,
    from: best.m.from,
    to: best.m.to,
    promotion: best.m.promotion,
    score: best.reply,
    depth: 2,
    note: `${persona.label} · ${modeLabel} · ${best.reply >= 0 ? '+' : ''}${best.reply.toFixed(2)} white`,
  };
}
