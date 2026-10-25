import { cpSync, mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { bibtexFromCff, computeFacts, PRESET_FILES, REPO_ROOT } from './sync-data.mjs';

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
		expect(facts.presets.map((p) => p.name)).toEqual(['Home', 'Genesis', 'First Light', 'Hunger', 'Roche', 'Undertow']);
		expect(by.champion.label).toBe('Our universe. The strongest so far (Gen 4).');
		expect(by['tidal-load'].label).toBe(
			"Enemy pull drags on the king's whole body. Rejected, it made the engine weaker. Home's constants with this law switched on."
		);
		expect(by['equal-start'].selfPlayGames).toBe(60000);
		expect(by['gen-0'].selfPlayGames).toBe(0);
		expect(facts.presets.filter((p) => p.variant).map((p) => p.slug)).toEqual(['accretion', 'strong-roche', 'tidal-load']);
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

	it('reads the proof line (Gen 3 vs Stockfish 19 at 2100) and the ledger cards', () => {
		const { proof, ledger } = facts.landing;
		expect(proof).toEqual({
			generation: 3,
			opponent: 'Stockfish 19',
			setting: 2100,
			baseSeconds: 30,
			incrementSeconds: 0.3,
			games: 200,
			w: 103,
			d: 5,
			l: 92,
			scorePercent: 53,
			source: 'results/ladder-gen003.json'
		});
		expect(ledger.map((c) => [c.id, c.verdict])).toEqual([
			['field', 'kept'],
			['space', 'kept'],
			['binding', 'back'],
			['accretion', 'rejected'],
			['king', 'rejected-twice']
		]);
		for (const c of ledger) for (const e of c.evidence) expect(existsSync(join(engine, e))).toBe(true);
		expect(Object.fromEntries(ledger.map((c) => [c.id, c.universes]))).toEqual({
			field: [],
			space: [],
			binding: [],
			accretion: ['accretion'],
			king: ['strong-roche', 'tidal-load']
		});
	});
});

describe('citation', () => {
	const cffText = readFileSync(join(REPO_ROOT, 'CITATION.cff'), 'utf8');

	it('builds the BibTeX from CITATION.cff: title, author and url, nothing else', () => {
		const cite = computeFacts().cite;
		const title = /^title:\s*"(.+)"$/m.exec(cffText)[1];
		const url = /^url:\s*"(.+)"$/m.exec(cffText)[1];
		expect(cite.title).toBe(title);
		expect(cite.url).toBe(url);
		expect(cite.authors).toEqual(['Baruah, Ripuranjan']);
		expect(cite.bibtex).toContain(`title  = {${title}}`);
		expect(cite.bibtex).toContain(`url    = {${url}}`);
		expect(cite.bibtex).toMatch(/^@software\{/);
		expect(cite.bibtex).not.toMatch(/version|doi|year/i);
	});

	it('fails when a needed field is missing', () => {
		expect(() => bibtexFromCff(cffText.replace(/^title:.*$/m, ''))).toThrow(/no title/);
		expect(() => bibtexFromCff(cffText.replace(/^url:.*$/m, ''))).toThrow(/no url/);
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
			rmSync(join(dir, 'results/ladder-gen003.json'));
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
