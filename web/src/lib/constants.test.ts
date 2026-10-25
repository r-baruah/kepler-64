import { describe, expect, it } from 'vitest';
import { constantRows, diffLines, fmtConstant } from './constants';
import type { Constants } from './types';

const C: Constants = {
	alpha: 0.5393913984298706, beta: 2.7456345558166504, c: 2, f_acc: 0, k_roche: 3.33, kappa: 2.6252963542938232,
	lambda_bind: 0.019230974838137627, lambda_space: 0.12638120353221893, lambda_tide: 0, m_b: 3.1232614517211914,
	m_n: 2.7217466831207275, m_q: 7.386493682861328, m_r: 4.241305828094482, rho: 1.56, sigma: 0.5765364766120911
};

describe('constants plate', () => {
	it('formats values plainly', () => {
		expect(fmtConstant('m_q', 7.386493682861328)).toBe('7.39');
		expect(fmtConstant('c', 2)).toBe('2 moves');
		expect(fmtConstant('f_acc', 0)).toBe('0');
		expect(fmtConstant('lambda_bind', 0.019230974838137627)).toBe('0.019');
		expect(fmtConstant('m_r', 5)).toBe('5');
	});
	it('never shows the Roche constants for the champion', () => {
		const keys = constantRows({ slug: 'champion', constants: C }).map((r) => r.key);
		expect(keys).not.toContain('beta');
		expect(keys).not.toContain('rho');
		expect(keys).not.toContain('k_roche');
		expect(keys).toContain('c');
	});
	it('shows the Roche term only for the strong Roche preset', () => {
		expect(constantRows({ slug: 'strong-roche', constants: { ...C, beta: 10 } }).map((r) => r.key)).toContain('beta');
	});
	it('names differences in plain words and counts the unnamed ones', () => {
		const d = diffLines({
			slug: 'gen-0',
			diff: { m_q: { champion: 7.386, preset: 9 }, rho: { champion: 1.56, preset: 1.5 }, beta: { champion: 2.75, preset: 2 } }
		});
		expect(d).toEqual({ lines: ['Queen 7.4 → 9.0'], more: 2 });
		expect(diffLines({ slug: 'accretion', diff: { f_acc: { champion: 0, preset: 0.8 } } })).toEqual({ lines: ['Captures absorb weight: on'], more: 0 });
		expect(diffLines({ slug: 'tidal-load', diff: { lambda_tide: { champion: 0, preset: 0.1 } } }).lines).toEqual(['Enemy pull drags on the king: on']);
		expect(diffLines({ slug: 'strong-roche', diff: { beta: { champion: 2.75, preset: 10 } } }).lines).toEqual(['Roche limit turned up: 2.75 → 10']);
		expect(diffLines({ slug: 'champion', diff: {} })).toEqual({ lines: [], more: 0 });
		// A line the fixed label already says is not repeated.
		expect(diffLines({ slug: 'accretion', label: 'Captures absorb weight: a piece grows by eating.', diff: { f_acc: { champion: 0, preset: 0.8 } } })).toEqual({ lines: [], more: 0 });
		const all = diffLines({
			slug: 'gen-0',
			diff: { m_n: { champion: 2.72, preset: 3 }, m_b: { champion: 3.12, preset: 3 }, m_r: { champion: 4.24, preset: 5 }, m_q: { champion: 7.39, preset: 9 } }
		});
		expect(all).toEqual({ lines: ['Weights N 3.0, B 3.0, R 5.0, Q 9.0'], more: 0 });
	});
});
