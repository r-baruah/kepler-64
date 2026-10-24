import { describe, expect, it } from 'vitest';
import fx from './fixtures/breakdowns.json';
import { bodyRadius, fieldDots, pieceWeights } from './field';
import type { Breakdown } from './types';

const F = fx as unknown as Record<string, Breakdown>;

function flat(white: Record<number, number>, black: Record<number, number>) {
	const w = new Array(64).fill(0);
	const b = new Array(64).fill(0);
	for (const [k, v] of Object.entries(white)) w[Number(k)] = v;
	for (const [k, v] of Object.entries(black)) b[Number(k)] = v;
	return { field: { white: w, black: b } };
}

describe('fieldDots', () => {
	it('draws nothing under the threshold, a solid dot for White, a ring for Black', () => {
		const dots = fieldDots(flat({ 0: 0.59, 1: 0.6, 2: 6 }, { 3: 3, 4: 0.5 }));
		expect(dots.map((d) => [d.sq, d.kind])).toEqual([
			[1, 'solid'],
			[2, 'solid'],
			[3, 'ring']
		]);
	});
	it('sizes between 12% and 32% of the square and caps at 6', () => {
		const [small, big, capped] = fieldDots(flat({ 0: 0.6, 1: 6, 2: 60 }, {}));
		expect(small.size).toBeCloseTo(0.12 + 0.2 * 0.1, 5);
		expect(big.size).toBeCloseTo(0.32, 5);
		expect(capped.size).toBeCloseTo(0.32, 5);
		for (const d of [small, big, capped]) expect(d.opacity).toBeGreaterThanOrEqual(0.22);
		expect(capped.opacity).toBeLessThanOrEqual(0.3 + 1e-9);
	});
	it('uses the engine field for the start position (both sides present)', () => {
		const dots = fieldDots(F.start);
		expect(dots.some((d) => d.kind === 'solid' && d.sq < 32)).toBe(true);
		expect(dots.some((d) => d.kind === 'ring' && d.sq >= 32)).toBe(true);
	});
});

describe('piece weights', () => {
	it('gives area proportional to weight, pawn = 1', () => {
		const w = pieceWeights(F.start.constants);
		expect(w[0].weight).toBe(1);
		const rPawn = bodyRadius(1, 6);
		const rQueen = bodyRadius(w[4].weight, 6);
		expect((rQueen * rQueen) / (rPawn * rPawn)).toBeCloseTo(w[4].weight, 9);
	});
});
