import { describe, expect, it } from 'vitest';
import { parsePlacement, sanFor, sqIndex, sqName, START_FEN, withSuffix } from './chess';

describe('squares', () => {
	it('maps names and indices both ways', () => {
		expect(sqIndex('a1')).toBe(0);
		expect(sqIndex('h8')).toBe(63);
		expect(sqName(sqIndex('e4'))).toBe('e4');
	});
	it('reads the placement', () => {
		const b = parsePlacement(START_FEN);
		expect(b[sqIndex('e1')]).toEqual({ piece: 'K', color: 'white' });
		expect(b[sqIndex('d8')]).toEqual({ piece: 'Q', color: 'black' });
		expect(b[sqIndex('e4')]).toBeNull();
	});
});

describe('san', () => {
	it('writes pawn, piece, capture and castling moves', () => {
		expect(sanFor(START_FEN, 'e2e4', ['e2e4'])).toBe('e4');
		expect(sanFor(START_FEN, 'g1f3', ['g1f3', 'b1c3'])).toBe('Nf3');
		const cap = 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
		expect(sanFor(cap, 'e4d5', ['e4d5'])).toBe('exd5');
		const castle = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1';
		expect(sanFor(castle, 'e1g1', ['e1g1', 'e1c1'])).toBe('O-O');
		expect(sanFor(castle, 'e1c1', ['e1g1', 'e1c1'])).toBe('O-O-O');
	});
	it('disambiguates by file, then rank, and writes promotions', () => {
		const rooks = '4k3/8/8/8/8/8/8/R3K2R w - - 0 1';
		expect(sanFor(rooks, 'a1d1', ['a1d1', 'h1d1'])).toBe('Rad1');
		const stacked = '4k3/R7/8/8/8/8/R7/4K3 w - - 0 1';
		expect(sanFor(stacked, 'a2a4', ['a2a4', 'a7a4'])).toBe('R2a4');
		expect(sanFor('8/P6k/8/8/8/8/8/K7 w - - 0 1', 'a7a8q', ['a7a8q'])).toBe('a8=Q');
	});
	it('adds check and mate suffixes', () => {
		expect(withSuffix('Qh5', true, false)).toBe('Qh5+');
		expect(withSuffix('Ra8', true, true)).toBe('Ra8#');
		expect(withSuffix('e4', false, false)).toBe('e4');
	});
});
