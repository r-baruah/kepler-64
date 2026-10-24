// Turns the engine's two 64-value fields into the dot plate: a solid dot where White pulls harder,
// a ring where Black does, sized by the difference. Pure; the board component only draws the result.
import type { Breakdown, Constants } from './types';

/** Differences smaller than this draw nothing. */
export const FIELD_THRESHOLD = 0.6;
/** Differences at or above this draw the largest dot. */
export const FIELD_CAP = 6;

export interface Dot {
	/** Square index, a1 = 0. */
	sq: number;
	kind: 'solid' | 'ring';
	/** Dot diameter as a fraction of the square: 0.12 (smallest) to 0.32 (largest). */
	size: number;
	/** 0.22 to 0.30. */
	opacity: number;
}

export function fieldDots(b: Pick<Breakdown, 'field'>): Dot[] {
	const out: Dot[] = [];
	for (let i = 0; i < 64; i++) {
		const d = b.field.white[i] - b.field.black[i];
		if (Math.abs(d) < FIELD_THRESHOLD) continue;
		const s = Math.min(Math.abs(d), FIELD_CAP) / FIELD_CAP;
		out.push({ sq: i, kind: d > 0 ? 'solid' : 'ring', size: 0.12 + 0.2 * s, opacity: 0.22 + 0.08 * s });
	}
	return out;
}

export interface WeightBody {
	letter: 'P' | 'N' | 'B' | 'R' | 'Q';
	name: string;
	weight: number;
}

/** The learned weights, pawn = 1, in the order they are drawn. */
export function pieceWeights(c: Pick<Constants, 'm_n' | 'm_b' | 'm_r' | 'm_q'>): WeightBody[] {
	return [
		{ letter: 'P', name: 'Pawn', weight: 1 },
		{ letter: 'N', name: 'Knight', weight: c.m_n },
		{ letter: 'B', name: 'Bishop', weight: c.m_b },
		{ letter: 'R', name: 'Rook', weight: c.m_r },
		{ letter: 'Q', name: 'Queen', weight: c.m_q }
	];
}

/** Circle radius for a weight: area proportional to weight, so radius goes with the square root. */
export function bodyRadius(weight: number, pawnRadius: number): number {
	return pawnRadius * Math.sqrt(weight);
}

export function oneDecimal(n: number): string {
	return n.toFixed(1);
}
