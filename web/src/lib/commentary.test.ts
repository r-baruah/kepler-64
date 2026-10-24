import { describe, expect, it } from 'vitest';
import { commentary } from './commentary';
import fx from './fixtures/breakdowns.json';
import type { Breakdown, PieceReport } from './types';

const F = fx as unknown as Record<string, Breakdown>;

/** Build a small synthetic breakdown from a list of pieces (only fields the commentary reads). */
function bd(pieces: Partial<PieceReport>[]): Breakdown {
	const full = pieces.map((p) => ({
		piece: 'P',
		color: 'white',
		rest_mass: 1,
		accreted: 0,
		carried_mass: 1,
		emission: 1,
		energy: 0,
		hops: [[], [], []],
		...p
	})) as PieceReport[];
	return { pieces: full } as unknown as Breakdown;
}

describe('commentary on real engine output (gen-004)', () => {
	it('describes Kepler\'s ...Nc6 with counts taken from the report', () => {
		const text = commentary(F.afterNf3, F.afterNc6, { uci: 'b8c6', mover: 'kepler' });
		const c6 = F.afterNc6.pieces.find((p) => p.square === 'c6')!;
		const b8 = F.afterNf3.pieces.find((p) => p.square === 'b8')!;
		expect(c6.hops[0]).toHaveLength(8);
		expect(text).toBe(
			`Kepler's knight now pulls on 8 squares (${b8.hops[0].length} before), including d8, where it defends its queen.`
		);
	});

	it('uses "Your" for the human side and a slider block with an own piece', () => {
		const text = commentary(F.start, F.afterA4, { uci: 'a2a4', mover: 'you' });
		expect(text).toBe('Your pawn now pulls on 1 square.');
	});

	it('says nothing when the moved piece cannot be found', () => {
		expect(commentary(F.start, F.afterNc6, { uci: 'e4e5', mover: 'you' })).toBe('');
	});

	it('handles castling (the king is the moved piece) and promotion (a new piece)', () => {
		const castle = commentary(F.castleBefore, F.castleAfter, { uci: 'e1g1', mover: 'you' });
		expect(castle).toMatch(/^Your king now pulls on \d+ squares?/);
		const promo = commentary(F.promoBefore, F.promoAfter, { uci: 'a7a8q', mover: 'kepler' });
		expect(promo).toMatch(/^Kepler's new queen now pulls on \d+ squares?/);
		expect(promo).not.toContain('before');
	});
});

describe('commentary templates (synthetic)', () => {
	it('pluralises and handles zero', () => {
		const before = bd([{ square: 'b1', piece: 'N', hops: [['a3', 'c3'], [], []] }]);
		const one = bd([{ square: 'a3', piece: 'N', hops: [['b1'], [], []] }]);
		expect(commentary(before, one, { uci: 'b1a3', mover: 'you' })).toBe('Your knight now pulls on 1 square (2 before).');
		const none = bd([{ square: 'a3', piece: 'N', hops: [[], [], []] }]);
		expect(commentary(before, none, { uci: 'b1a3', mover: 'kepler' })).toBe("Kepler's knight now pulls on no squares (2 before).");
	});

	it('names the heaviest defended own piece for a knight', () => {
		const before = bd([{ square: 'b1', piece: 'N', hops: [['a3'], [], []] }]);
		const after = bd([
			{ square: 'c3', piece: 'N', hops: [['a2', 'b1', 'd1'], [], []] },
			{ square: 'd1', piece: 'Q', rest_mass: 7 },
			{ square: 'a2', piece: 'P' },
			{ square: 'b1', piece: 'R', rest_mass: 4 }
		]);
		expect(commentary(before, after, { uci: 'b1c3', mover: 'kepler' })).toBe(
			"Kepler's knight now pulls on 3 squares (1 before), including d1, where it defends its queen."
		);
	});

	it('claims a block when a slider ray ends on its own piece with board beyond it', () => {
		const prev = bd([{ square: 'd1', piece: 'R', hops: [['d2'], [], []] }]);
		const after = bd([
			{ square: 'a1', piece: 'R', hops: [['a2', 'a3', 'b1'], [], []] },
			{ square: 'a3', piece: 'P' },
			{ square: 'b1', piece: 'N', rest_mass: 3 }
		]);
		// The a-file ray ends at a3 (a4 lies beyond) and the rank ray ends at b1 (c1 lies beyond).
		// The knight is heavier than the pawn, so it is the one named.
		expect(commentary(prev, after, { uci: 'd1a1', mover: 'you' })).toBe(
			'Your rook now pulls on 3 squares (1 before), blocked at b1 by its own knight.'
		);
	});

	it('does not claim a block at the edge of the board', () => {
		const prev = bd([{ square: 'a2', piece: 'R', hops: [['a3'], [], []] }]);
		const edge = bd([
			{ square: 'a1', piece: 'R', hops: [['b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1'], [], []] },
			{ square: 'h1', piece: 'N' }
		]);
		expect(commentary(prev, edge, { uci: 'a2a1', mover: 'you' })).not.toContain('blocked');
	});

	it('falls back to an enemy piece in reach, naming the other side', () => {
		const prev = bd([{ square: 'd3', piece: 'B', hops: [[], [], []] }]);
		const after = bd([
			{ square: 'e4', piece: 'B', hops: [['h7'], [], []] },
			{ square: 'h7', piece: 'P', color: 'black' }
		]);
		expect(commentary(prev, after, { uci: 'd3e4', mover: 'kepler' })).toBe(
			"Kepler's bishop now pulls on 1 square (0 before), including h7, where your pawn stands."
		);
	});
});
