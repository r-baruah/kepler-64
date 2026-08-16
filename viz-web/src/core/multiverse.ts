/**
 * Kepler-64 Multiverse Sampler
 * Samples alternate physical universes around a base constant set and evaluates
 * a position across them, returning mean score and Bayesian volatility (sigma).
 * Shared by the Quantum Multiverse bot (2.1) and the uncertainty ribbon (3.1).
 */

import type { ConstantsConfig } from './constants';
import { KeplerBoard } from './board';
import { evaluatePosition } from './evaluate';

export interface UniverseSample {
  config: ConstantsConfig;
  label: string;
  weight: number;
}

const UNIVERSE_LABELS = [
  'α High-Mass',
  'β Fast-Light',
  'γ Soft-Potential',
  'δ Tight-Roche',
  'ε Slow-Light',
  'ζ Hard-Mass',
  'η Loose-Roche',
  'θ Cold-Field',
];

function gaussian(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function sampleUniverses(base: ConstantsConfig, k: number): UniverseSample[] {
  const samples: UniverseSample[] = [];
  for (let i = 0; i < k; i++) {
    const perturb = () => 1.0 + gaussian() * 0.16;
    const config: ConstantsConfig = {
      ...base,
      G: clamp(base.G * perturb(), 0.3, 4.0),
      eps: clamp(base.eps * perturb(), 0.3, 2.5),
      c: clamp(base.c * perturb(), 1.0, 10.0),
      roche: clamp(base.roche * perturb(), 0.3, 1.6),
      mat_gain: clamp(base.mat_gain * perturb(), 0.4, 2.2),
    };
    samples.push({
      config,
      label: UNIVERSE_LABELS[i % UNIVERSE_LABELS.length],
      weight: 1 / k,
    });
  }
  return samples;
}

export interface MultiverseScore {
  mean: number;
  sigma: number;
  scores: number[];
  samples: UniverseSample[];
}

/**
 * Evaluates a FEN across sampled universes.
 * Returns white-perspective mean and population standard deviation.
 */
export function evaluateAcrossUniverses(
  fen: string,
  samples: UniverseSample[],
  boost?: Float32Array
): MultiverseScore {
  const scores = samples.map((s) => {
    const board = new KeplerBoard(fen);
    if (boost) board.massBoost = boost;
    return evaluatePosition(board, s.config).totalScoreWhite;
  });

  const n = scores.length || 1;
  const mean = scores.reduce((a, b) => a + b, 0) / n;
  const variance = scores.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  const sigma = Math.sqrt(Math.max(0, variance));

  return { mean, sigma, scores, samples };
}
