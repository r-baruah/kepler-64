// Plain names for the constants a visitor sees, with their meanings from docs/v1/02_LAWS_SPEC.md §2.
// Only constants with a literal meaning are listed. The Roche constants (rho, k_roche, beta) are shown
// only for the preset that switches that law up, so the champion's page never suggests a Roche limit
// drives its play (claims register).
import type { Constants, PresetFacts } from './types';

export interface ConstantRow {
	key: keyof Constants;
	name: string;
	/** One short phrase, true to the spec. */
	meaning: string;
	value: string;
}

type Key = keyof Constants;

const INFO: Partial<Record<Key, { name: string; meaning: string }>> = {
	m_n: { name: 'Knight weight', meaning: 'pawn = 1' },
	m_b: { name: 'Bishop weight', meaning: 'pawn = 1' },
	m_r: { name: 'Rook weight', meaning: 'pawn = 1' },
	m_q: { name: 'Queen weight', meaning: 'pawn = 1' },
	c: { name: 'Light speed', meaning: 'the most moves pull can travel' },
	alpha: { name: 'Pull kept per extra move', meaning: 'fraction left after each further move' },
	sigma: { name: 'Spreading', meaning: '0 = no thinning, 1 = pull shared over the squares reached' },
	kappa: { name: "King's own pull", meaning: 'how hard the king itself pulls' },
	lambda_space: { name: 'Held space', meaning: 'weight of squares owned by the stronger pull' },
	lambda_bind: { name: 'Binding', meaning: 'weight of defenders holding pieces together' },
	f_acc: { name: 'Accretion', meaning: "share of a captured piece's weight the captor absorbs (0 = off)" },
	lambda_tide: { name: 'Tidal load', meaning: "weight of enemy pull on the king's body (0 = off)" },
	beta: { name: 'Roche term', meaning: 'weight of a king past its breaking point' }
};

/** Keys in display order. */
const MAIN: Key[] = ['m_n', 'm_b', 'm_r', 'm_q', 'c', 'alpha', 'sigma', 'kappa', 'lambda_space', 'lambda_bind', 'f_acc', 'lambda_tide'];

export function fmtConstant(key: Key, n: number): string {
	if (key === 'c') return `${n} moves`;
	if (n === 0) return '0';
	const a = Math.abs(n);
	const s = a >= 10 ? n.toFixed(1) : a >= 1 ? n.toFixed(2) : a >= 0.1 ? n.toFixed(2) : n.toFixed(3);
	return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

/** Rows for the constants plate of a preset. */
export function constantRows(p: Pick<PresetFacts, 'slug' | 'constants'>): ConstantRow[] {
	const keys = [...MAIN];
	if (p.slug === 'strong-roche') keys.push('beta');
	return keys.map((key) => ({ key, ...INFO[key]!, value: fmtConstant(key, p.constants[key]) }));
}

const WEIGHT_NAMES: [Key, string, string][] = [
	['m_n', 'Knight', 'N'],
	['m_b', 'Bishop', 'B'],
	['m_r', 'Rook', 'R'],
	['m_q', 'Queen', 'Q']
];

/** Law switches named in plain words. `on` reads the preset's value; the champion has each one off or low. */
const LAW_WORDS: Partial<Record<Key, (from: number, to: number) => string>> = {
	f_acc: (a, b) => `Captures absorb weight: ${b > 0 ? 'on' : 'off'}` + (a > 0 && b > 0 ? ` (${fmtConstant('f_acc', a)} → ${fmtConstant('f_acc', b)})` : ''),
	lambda_tide: (a, b) => `Enemy pull drags on the king: ${b > 0 ? 'on' : 'off'}` + (a > 0 && b > 0 ? ` (${fmtConstant('lambda_tide', a)} → ${fmtConstant('lambda_tide', b)})` : ''),
	c: (a, b) => `Light speed ${a} → ${b} moves`
};

/**
 * What differs from Home, in plain words: one line per changed piece weight ("Queen 7.4 → 9.0", or one
 * line of this universe's weights when three or more differ), one
 * per law switch, and a count of the other constants that differ (shown in full behind the
 * constants disclosure). The Roche term is named only for the preset that turns it up.
 */
/** Phrases in a preset's fixed label that already say a law is switched on (so the line is not repeated). */
const SAID_BY_LABEL: Partial<Record<Key, string[]>> = {
	f_acc: ['captures absorb weight', 'accretion on'],
	lambda_tide: ['enemy pull drags']
};

export function diffLines(p: Pick<PresetFacts, 'slug' | 'diff'> & { label?: string }): { lines: string[]; more: number } {
	const label = (p.label ?? '').toLowerCase();
	const lines: string[] = [];
	let used = 0;
	const changed = WEIGHT_NAMES.filter(([k]) => k in p.diff);
	if (changed.length >= 3) {
		// Most weights differ: one line with this universe's weights (the drawing shows them too).
		lines.push('Weights ' + changed.map(([k, , letter]) => `${letter} ${p.diff[k].preset.toFixed(1)}`).join(', '));
	} else {
		for (const [k, name] of changed) lines.push(`${name} ${p.diff[k].champion.toFixed(1)} → ${p.diff[k].preset.toFixed(1)}`);
	}
	used += changed.length;
	for (const k of ['c', 'f_acc', 'lambda_tide'] as Key[]) {
		if (!(k in p.diff)) continue;
		used++;
		if (SAID_BY_LABEL[k]?.some((s) => label.includes(s))) continue;
		lines.push(LAW_WORDS[k]!(p.diff[k].champion, p.diff[k].preset));
	}
	if (p.slug === 'strong-roche' && 'beta' in p.diff) {
		lines.push(`Roche limit turned up: ${fmtConstant('beta', p.diff.beta.champion)} → ${fmtConstant('beta', p.diff.beta.preset)}`);
		used++;
	}
	return { lines, more: Object.keys(p.diff).length - used };
}
