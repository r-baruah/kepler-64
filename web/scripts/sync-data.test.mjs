import { cpSync, mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { computeFacts, PRESET_FILES, REPO_ROOT } from './sync-data.mjs';

const plan = join(REPO_ROOT, 'docs/v1/06_WEB_PLAN.md');
const engine = join(REPO_ROOT, 'engine');

describe('site facts', () => {
	const facts = computeFacts();

	it('sums self-play along the champion lineage (5,500 + 40,000 + 40,000)', () => {
		expect(facts.selfPlayGames).toBe(85500);
		expect(facts.lineage.filter((l) => l.games > 0).map((l) => [l.generation, l.games])).toEqual([
			[4, 40000],
			[3, 40000],
			[1, 5500]
		]);
	});

	it('reads the six presets with their fixed labels from the web plan', () => {
		expect(facts.presets.map((p) => p.slug)).toEqual(Object.keys(PRESET_FILES));
		const by = Object.fromEntries(facts.presets.map((p) => [p.slug, p]));
		expect(by.champion.label).toBe('The current strongest universe.');
		expect(by['tidal-load'].label).toBe(
			"Enemy pull on the king's body costs points. Lost 77 Elo at Gen 3. Untrained variant of Gen 4."
		);
		expect(by['equal-start'].selfPlayGames).toBe(60000);
		expect(by['gen-0'].selfPlayGames).toBe(0);
	});

	it('lists only the constants that differ from the champion', () => {
		const by = Object.fromEntries(facts.presets.map((p) => [p.slug, p]));
		expect(Object.keys(by.champion.diff)).toEqual([]);
		expect(Object.keys(by.accretion.diff)).toEqual(['f_acc']);
		expect(Object.keys(by['strong-roche'].diff)).toEqual(['beta']);
		expect(Object.keys(by['tidal-load'].diff)).toEqual(['lambda_tide']);
		expect(by['tidal-load'].diff.lambda_tide).toEqual({ champion: 0, preset: 0.1 });
	});

	it('reads the three levels as node budgets', () => {
		expect(facts.levels.map((l) => [l.id, l.nodes])).toEqual([
			['quick', 60000],
			['default', 400000],
			['strong', 1500000]
		]);
	});

	it('reads the piece-weight chart and the ledger cards', () => {
		const { chart, found, ledger } = facts.landing;
		expect(chart.runs.a).toHaveLength(31);
		expect(chart.runs.b).toHaveLength(31);
		expect(chart.start).toEqual([4.4286, 4.4286, 4.4286, 4.4286]);
		expect(chart.runs.a[0]).toEqual(chart.start);
		expect(found).toEqual({ n: 2.6, b: 3.3, r: 4.4, q: 8.2 });
		expect(chart.games).toBe(60000);
		expect(ledger.map((c) => [c.id, c.verdict])).toEqual([
			['field', 'kept'],
			['space', 'kept'],
			['binding', 'back'],
			['accretion', 'rejected'],
			['king', 'rejected-twice']
		]);
		for (const c of ledger) for (const e of c.evidence) expect(existsSync(join(engine, e))).toBe(true);
	});
});

describe('missing sources fail the build', () => {
	it('throws when a results file or a preset file is missing', () => {
		const dir = mkdtempSync(join(tmpdir(), 'kepler-sync-'));
		try {
			cpSync(join(engine, 'data'), join(dir, 'data'), { recursive: true });
			cpSync(join(engine, 'results'), join(dir, 'results'), { recursive: true });
			expect(() => computeFacts(dir, plan)).not.toThrow();
			rmSync(join(dir, 'results/gen003-tdleaf-a/candidate.json'));
			expect(() => computeFacts(dir, plan)).toThrow(/missing source file/);
			cpSync(join(engine, 'results/gen003-tdleaf-a/candidate.json'), join(dir, 'results/gen003-tdleaf-a/candidate.json'));
			rmSync(join(dir, 'data/ablations/gen004-tidal.json'));
			expect(() => computeFacts(dir, plan)).toThrow(/missing source file/);
			cpSync(join(engine, 'data/ablations/gen004-tidal.json'), join(dir, 'data/ablations/gen004-tidal.json'));
			rmSync(join(dir, 'results/ledger-gen003-bind0.json'));
			expect(() => computeFacts(dir, plan)).toThrow(/missing source file/);
			cpSync(join(engine, 'results/ledger-gen003-bind0.json'), join(dir, 'results/ledger-gen003-bind0.json'));
			rmSync(join(dir, 'results/control-equal-masses-b/round-017.json'));
			expect(() => computeFacts(dir, plan)).toThrow(/missing source file/);
		} finally {
			rmSync(dir, { recursive: true, force: true });
		}
	});

	it('writes site-facts.json that matches computeFacts', () => {
		const file = join(REPO_ROOT, 'web/static/data/site-facts.json');
		if (existsSync(file)) expect(JSON.parse(readFileSync(file, 'utf8')).selfPlayGames).toBe(85500);
	});
});
