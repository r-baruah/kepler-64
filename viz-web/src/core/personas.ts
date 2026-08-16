/**
 * Kepler-64 Engine Personas for Interactive Bot Mode (2.1)
 */

import type { ConstantsConfig } from './constants';
import { DEFAULT_CONSTANTS } from './constants';

export type SearchStyle = 'classic' | 'quantum';

export interface BotPersona {
  id: string;
  label: string;
  blurb: string;
  config: ConstantsConfig;
  search: SearchStyle;
  beam: number;         // Number of top-level candidates carried into 2-ply refinement
  quantumSamples: number;
  riskLambda: number;   // Bayesian risk-aversion coefficient (mean - lambda * sigma)
}

export const BOT_PERSONAS: BotPersona[] = [
  {
    id: 'newtonian',
    label: 'Newtonian Classicist',
    blurb: 'High G and a tight Roche radius — brute-force mass accumulation.',
    config: { ...DEFAULT_CONSTANTS, G: 2.2, roche: 0.62, mat_gain: 1.25, accEta: 0.80 },
    search: 'classic',
    beam: 8,
    quantumSamples: 0,
    riskLambda: 0,
  },
  {
    id: 'relativistic',
    label: 'Relativistic Attacker',
    blurb: 'Fast light gate and a capture-hungry material appetite — feeds the Queen into a supermassive well.',
    config: { ...DEFAULT_CONSTANTS, c: 9.0, roche: 0.55, mat_gain: 1.6, accEta: 0.95 },
    search: 'classic',
    beam: 8,
    quantumSamples: 0,
    riskLambda: 0,
  },
  {
    id: 'quantum',
    label: 'Quantum Multiverse Bot',
    blurb: 'Samples 5 alternate universes per move and chooses the minimax line with highest Bayesian expectation.',
    config: { ...DEFAULT_CONSTANTS },
    search: 'quantum',
    beam: 5,
    quantumSamples: 5,
    riskLambda: 0.5,
  },
];

export function getPersona(id: string): BotPersona {
  return BOT_PERSONAS.find((p) => p.id === id) ?? BOT_PERSONAS[0];
}
