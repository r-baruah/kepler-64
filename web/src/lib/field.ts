// Turns the engine's two 64-value fields into what the board draws. Pure; the board only renders.
//
// The field is drawn as square tints, never as dots or rings:
// dots are reserved for legal-move hints, so the two can't be confused.
import type { Breakdown, Constants } from './types';

/** Differences smaller than this draw nothing. */
export const FIELD_THRESHOLD = 0.6;
/** Differences at or above this draw at full strength. */
export const FIELD_CAP = 6;

export interface Tint {
	/** Square index, a1 = 0. */
	sq: number;
	/** Which side pulls harder on this square. */
	side: 'white' | 'black';
	/** 0 (just over the threshold) to 1 (at or over the cap). */
	strength: number;
}

/** One tint per square where |Φ_W − Φ_B| reaches the threshold, scaled by the difference. */
export function fieldTints(b: Pick<Breakdown, 'field'>): Tint[] {
	const out: Tint[] = [];
	for (let i = 0; i < 64; i++) {
		const d = b.field.white[i] - b.field.black[i];
		const a = Math.abs(d);
		if (a < FIELD_THRESHOLD) continue;
		const strength = (Math.min(a, FIELD_CAP) - FIELD_THRESHOLD) / (FIELD_CAP - FIELD_THRESHOLD);
		out.push({ sq: i, side: d > 0 ? 'white' : 'black', strength });
	}
	return out;
}

/** Contour levels of Φ_W − Φ_B (positive: White pulls harder). */
export const CONTOUR_LEVELS = [1.5, 3, 4.5];

export interface Contour {
	/** Signed level; positive for White. */
	level: number;
	/** SVG path in board units (800 x 800, White at the bottom; the caller mirrors it when flipped). */
	d: string;
}

/**
 * Contour lines of the field difference, as seen with White at the bottom. Values sit at square
 * centres and are interpolated smoothly between them (clamped at the edges), then traced with
 * marching squares. The lines are a drawing of the same 64 numbers the tints use, nothing more.
 */
export function fieldContours(b: Pick<Breakdown, 'field'>, levels: number[] = CONTOUR_LEVELS, step = 12.5): Contour[] {
	const v = (f: number, r: number) => {
		const i = r * 8 + f;
		return b.field.white[i] - b.field.black[i];
	};
	// Sample grid in board units: x from 0 to 800 (files), y from 0 to 800 (rank 8 at top).
	const n = Math.round(800 / step) + 1;
	const g: number[] = new Array(n * n);
	for (let gy = 0; gy < n; gy++) {
		for (let gx = 0; gx < n; gx++) {
			const fx = Math.min(7, Math.max(0, (gx * step) / 100 - 0.5));
			const ry = Math.min(7, Math.max(0, 7 - ((gy * step) / 100 - 0.5)));
			const f0 = Math.min(6, Math.floor(fx));
			const r0 = Math.min(6, Math.floor(ry));
			// Smoothstep weights: the same values at the centres, rounder lines between them.
			const sx = fx - f0;
			const sy = ry - r0;
			const tx = sx * sx * (3 - 2 * sx);
			const ty = sy * sy * (3 - 2 * sy);
			g[gy * n + gx] =
				v(f0, r0) * (1 - tx) * (1 - ty) + v(f0 + 1, r0) * tx * (1 - ty) + v(f0, r0 + 1) * (1 - tx) * ty + v(f0 + 1, r0 + 1) * tx * ty;
		}
	}
	const out: Contour[] = [];
	for (const L of [...levels.map((l) => -l), ...levels]) {
		let d = '';
		const lerp = (a: number, bb: number) => (L - a) / (bb - a);
		for (let gy = 0; gy < n - 1; gy++) {
			for (let gx = 0; gx < n - 1; gx++) {
				const a = g[gy * n + gx];
				const bb = g[gy * n + gx + 1];
				const c = g[(gy + 1) * n + gx + 1];
				const e = g[(gy + 1) * n + gx];
				const x = gx * step;
				const y = gy * step;
				// Edge crossings: top (a-bb), right (bb-c), bottom (e-c), left (a-e).
				const pts: [number, number][] = [];
				if ((a < L) !== (bb < L)) pts.push([x + lerp(a, bb) * step, y]);
				if ((bb < L) !== (c < L)) pts.push([x + step, y + lerp(bb, c) * step]);
				if ((e < L) !== (c < L)) pts.push([x + lerp(e, c) * step, y + step]);
				if ((a < L) !== (e < L)) pts.push([x, y + lerp(a, e) * step]);
				for (let k = 0; k + 1 < pts.length; k += 2) {
					d += `M${pts[k][0].toFixed(1)} ${pts[k][1].toFixed(1)}L${pts[k + 1][0].toFixed(1)} ${pts[k + 1][1].toFixed(1)}`;
				}
			}
		}
		if (d) out.push({ level: L, d });
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
