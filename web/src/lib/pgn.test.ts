import { describe, expect, it } from 'vitest';
import { buildPgn, fileDate, pgnDate, pgnFileName, type PgnInput } from './pgn';

const base: PgnInput = {
	sans: ['e4', 'e5', 'Nf3'],
	human: 'white',
	result: null,
	universe: 'Home',
	level: 'Default',
	date: new Date(2026, 9, 5, 14, 30)
};

describe('pgn', () => {
	it('writes the seven tags, custom tags and numbered moves for a short game', () => {
		const pgn = buildPgn({ ...base, engineCommit: 'abc123\n' });
		expect(pgn).toBe(
			[
				'[Event "Kepler-64 casual game"]',
				'[Site "https://kepler.ripuranjan.com"]',
				'[Date "2026.10.05"]',
				'[Round "-"]',
				'[White "You"]',
				'[Black "Kepler-64"]',
				'[Result "*"]',
				'[Universe "Home"]',
				'[Level "Default"]',
				'[EngineCommit "abc123"]',
				'',
				'1. e4 e5 2. Nf3 *',
				''
			].join('\n')
		);
	});
	it('ends a checkmate with its result', () => {
		const pgn = buildPgn({ ...base, sans: ['f3', 'e5', 'g4', 'Qh4#'], human: 'white', result: '0-1' });
		expect(pgn).toContain('[Result "0-1"]');
		expect(pgn).not.toContain('EngineCommit');
		expect(pgn.trimEnd().endsWith('1. f3 e5 2. g4 Qh4# 0-1')).toBe(true);
	});
	it('marks an unfinished game with *', () => {
		const pgn = buildPgn(base);
		expect(pgn).toContain('[Result "*"]');
		expect(pgn.trimEnd().endsWith('*')).toBe(true);
	});
	it('puts You on the black side when the human plays Black', () => {
		const pgn = buildPgn({ ...base, human: 'black', result: '1/2-1/2' });
		expect(pgn).toContain('[White "Kepler-64"]');
		expect(pgn).toContain('[Black "You"]');
		expect(pgn).toContain('[Result "1/2-1/2"]');
	});
	it('wraps move text at 80 characters', () => {
		const sans = Array.from({ length: 40 }, (_, i) => (i % 2 ? 'Nf6' : 'Nc3'));
		const text = buildPgn({ ...base, sans, result: '1-0' }).split('\n\n')[1].trimEnd();
		const lines = text.split('\n');
		expect(lines.length).toBeGreaterThan(1);
		expect(Math.max(...lines.map((l) => l.length))).toBeLessThanOrEqual(80);
		expect(text.replace(/\n/g, ' ').endsWith('1-0')).toBe(true);
	});
	it('adds SetUp and FEN for a custom start and numbers from it', () => {
		const fen = '6k1/5ppp/8/8/8/8/8/R3K3 b - - 0 7';
		const pgn = buildPgn({ ...base, sans: ['g6', 'Ra8#'], startFen: fen });
		expect(pgn).toContain('[SetUp "1"]');
		expect(pgn).toContain(`[FEN "${fen}"]`);
		expect(pgn.trimEnd().endsWith('7... g6 8. Ra8# *')).toBe(true);
	});
	it('formats dates', () => {
		expect(pgnDate(base.date)).toBe('2026.10.05');
		expect(fileDate(base.date)).toBe('2026-10-05');
		expect(pgnFileName(base.date)).toBe('kepler-64-2026-10-05.pgn');
	});
});
