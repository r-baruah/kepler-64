import { describe, expect, it } from 'vitest';
import fx from './fixtures/breakdowns.json';
import { bodyRadius, fieldContours, fieldTints, pieceWeights } from './field';
import type { Breakdown } from './types';

const F = fx as unknown as Record<string, Breakdown>;

function flat(white: Record<number, number>, black: Record<number, number>) {
	const w = new Array(64).fill(0);
	const b = new Array(64).fill(0);
	for (const [k, v] of Object.entries(white)) w[Number(k)] = v;
	for (const [k, v] of Object.entries(black)) b[Number(k)] = v;
	return { field: { white: w, black: b } };
}

describe('fieldTints', () => {
	it('draws nothing under the threshold; White where White pulls harder, Black where Black does', () => {
		const tints = fieldTints(flat({ 0: 0.59, 1: 0.6, 2: 6 }, { 3: 3, 4: 0.5 }));
		expect(tints.map((t) => [t.sq, t.side])).toEqual([
			[1, 'white'],
			[2, 'white'],
			[3, 'black']
		]);
	});
	it('scales strength from 0 at the threshold to 1 at the cap, and caps there', () => {
		const [low, mid, top, over] = fieldTints(flat({ 0: 0.6, 1: 3.3, 2: 6, 3: 60 }, {}));
		expect(low.strength).toBeCloseTo(0, 9);
		expect(mid.strength).toBeCloseTo(0.5, 9);
		expect(top.strength).toBeCloseTo(1, 9);
		expect(over.strength).toBeCloseTo(1, 9);
	});
	it('uses the difference, not either side alone', () => {
		expect(fieldTints(flat({ 5: 4 }, { 5: 3.8 }))).toEqual([]);
	});
	it('uses the engine field for the start position (both sides present)', () => {
		const tints = fieldTints(F.start);
		expect(tints.some((t) => t.side === 'white' && t.sq < 32)).toBe(true);
		expect(tints.some((t) => t.side === 'black' && t.sq >= 32)).toBe(true);
	});
});

describe('fieldContours', () => {
	it('draws nothing for an empty field', () => {
		expect(fieldContours(flat({}, {}))).toEqual([]);
	});
	it('draws a positive contour around a strong White square and none for Black', () => {
		const c = fieldContours(flat({ 27: 6 }, {}));
		expect(c.length).toBeGreaterThan(0);
		expect(c.every((x) => x.level > 0)).toBe(true);
		// d4 is file 3, rank 4: its centre is at (350, 450) in board units; the lines stay near it.
		const nums = c.flatMap((x) => x.d.match(/-?\d+(\.\d+)?/g)!.map(Number));
		for (let i = 0; i < nums.length; i += 2) {
			expect(Math.abs(nums[i] - 350)).toBeLessThanOrEqual(100);
			expect(Math.abs(nums[i + 1] - 450)).toBeLessThanOrEqual(100);
		}
	});
	it('gives Black negative levels', () => {
		const c = fieldContours(flat({}, { 40: 5 }));
		expect(c.length).toBeGreaterThan(0);
		expect(c.every((x) => x.level < 0)).toBe(true);
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
