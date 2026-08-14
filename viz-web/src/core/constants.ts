/**
 * Kepler-64: Astrophysical Chess Evaluator
 * Core Physics Constants and Piece Mass Table
 */

export interface ConstantsConfig {
  G: number;            // Gravitational constant
  eps: number;          // Plummer softening length (prevents force singularity)
  c: number;            // Speed of light reach gate (squares/ply)
  roche: number;        // Critical Roche disruption parameter
  bonus: number;        // Magnitude of king disruption force term
  kgain: number;        // Sharpness of disruption force-sigmoid
  gamma: number;        // Weight of global field-energy binding edge
  Rg: number;           // King radius of gyration (spatial extent)
  mref: number;         // Reference tidal-stress scale (~ minor piece)
  mat_gain: number;     // Material balance weight
  lambda_delta: number; // Delta-eta rate term
  com_gain: number;     // Center-of-mass advance delta
  inertia_gain: number; // Attack concentration delta
  entropy_gain: number; // Coordination entropy delta
}

export const DEFAULT_CONSTANTS: ConstantsConfig = {
  G: 1.32,
  eps: 0.71,
  c: 4.82,
  roche: 0.80,
  bonus: 220.0,
  kgain: 4.0,
  gamma: 0.25,
  Rg: 1.0,
  mref: 3.5,
  mat_gain: 1.0,
  lambda_delta: 1.0,
  com_gain: 25.0,
  inertia_gain: 8.0,
  entropy_gain: 2.5,
};

export const PIECE_MASSES: Record<string, number> = {
  p: 1.0,
  n: 3.0,
  b: 3.0,
  r: 5.0,
  q: 9.0,
  k: 1000.0,
};
