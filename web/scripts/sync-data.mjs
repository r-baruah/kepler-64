// Copies the data the site reads from engine/ into static/data/ and writes site-facts.json.
// Every number the UI prints that is not computed live comes from here.
// The build fails (non-zero exit) if any source file is missing or a fact cannot be derived.
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const WEB_ROOT = resolve(HERE, '..');
export const REPO_ROOT = resolve(WEB_ROOT, '..');

/** Preset slug -> source file under engine/data (web plan 5c). */
export const PRESET_FILES = {
	champion: 'generations/gen-004.json',
	'equal-start': 'ablations/control-equal-masses-a-candidate.json',
	'gen-0': 'generations/gen-000.json',
	accretion: 'ablations/gen004-accretion.json',
	'strong-roche': 'ablations/gen004-strong-roche.json',
	'tidal-load': 'ablations/gen004-tidal.json'
};

/** Results files that record the self-play of one generation (champion lineage). */
export const LINEAGE_RESULTS = {
	1: 'results/gen001-t1-candidate.json',
	3: 'results/gen003-tdleaf-a/candidate.json',
	4: 'results/gen004-tide-a/candidate.json'
};

/** Law Ledger cards on the landing. Each names the committed result files that back its verdict. */
export const LEDGER_CARDS = [
	{ id: 'field', verdict: 'kept', title: 'Pull follows the moves', note: 'Beats counting pieces alone, by far', evidence: ['results/ledger-gen001', 'results/ledger-gen001.md'] },
	{ id: 'space', verdict: 'kept', title: 'Squares belong to the stronger pull', note: 'Control of space', evidence: ['results/ledger-gen001.md'] },
	{ id: 'binding', verdict: 'back', title: 'Defenders hold pieces together', note: 'Cut at first, useful later', evidence: ['results/ledger-gen003-bind0.json'] },
	{ id: 'accretion', verdict: 'rejected', title: 'Captures absorb weight', note: 'Better switched off', evidence: ['results/ledger-gen001.md'] },
	{ id: 'king', verdict: 'rejected-twice', title: 'A king torn apart by pull', note: 'Two designs, neither helped', evidence: ['results/ledger-gen003-roche0.json', 'results/ledger-gen004tide-tide0.json'] }
];

/** The piece-weight chart: the equal-masses start, then 30 rounds of two independent runs. */
const CHART_RUNS = { a: 'results/control-equal-masses-a', b: 'results/control-equal-masses-b' };
const CHART_START = 'data/ablations/start-equal-masses.json';
const CHART_ROUNDS = 30;
const PIECES = ['n', 'b', 'r', 'q'];

function pieceMasses(c) {
	return PIECES.map((k) => {
		if (!Number.isFinite(c?.[`m_${k}`])) throw new Error(`sync-data: constant m_${k} missing`);
		return c[`m_${k}`];
	});
}

function computeLanding(engineRoot) {
	const start = pieceMasses(readJson(join(engineRoot, CHART_START)).constants);
	const runs = {};
	let games = 0;
	for (const [name, dir] of Object.entries(CHART_RUNS)) {
		const series = [start];
		for (let r = 1; r <= CHART_ROUNDS; r++) {
			const doc = readJson(join(engineRoot, dir, `round-${String(r).padStart(3, '0')}.json`));
			series.push(pieceMasses(doc.constants));
			if (name === 'a') {
				if (!Number.isFinite(doc.games)) throw new Error(`sync-data: round ${r} has no games`);
				games += doc.games;
			}
		}
		runs[name] = series;
	}
	const last = runs.a[CHART_ROUNDS];
	const found = Object.fromEntries(PIECES.map((k, i) => [k, Math.round(last[i] * 10) / 10]));
	const cards = LEDGER_CARDS.map((c) => {
		for (const e of c.evidence) need(join(engineRoot, e));
		return { ...c };
	});
	return { chart: { pieces: PIECES, start, runs, games }, found, ledger: cards };
}

const CHAMPION_FILE = 'generations/gen-004.json';

function need(path) {
	if (!existsSync(path)) throw new Error(`sync-data: missing source file ${path}`);
	return path;
}

function readJson(path) {
	return JSON.parse(readFileSync(need(path), 'utf8'));
}

/** Parse a markdown table that follows `heading`; returns rows as arrays of trimmed cells. */
function tableAfter(md, heading) {
	const at = md.indexOf(heading);
	if (at < 0) throw new Error(`sync-data: heading not found in web plan: ${heading}`);
	const lines = md.slice(at).split('\n').slice(1);
	const rows = [];
	let started = false;
	for (const line of lines) {
		if (line.trim().startsWith('|')) {
			started = true;
			rows.push(line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim()));
		} else if (started) break;
	}
	return rows.slice(2); // drop header and separator
}

const stripQuotes = (s) => s.replace(/^["“]/, '').replace(/["”]$/, '');
const stripTicks = (s) => s.replace(/`/g, '');

/** Constants with the optional lambda_tide made explicit (files omit it at 0). */
function normConstants(c) {
	return { ...c, lambda_tide: c.lambda_tide ?? 0 };
}

export function constantsDiff(base, other) {
	const out = {};
	for (const k of Object.keys(base)) {
		if (Math.abs(base[k] - other[k]) > 1e-9) out[k] = { champion: base[k], preset: other[k] };
	}
	return out;
}

export function computeFacts(engineRoot = join(REPO_ROOT, 'engine'), planPath = join(REPO_ROOT, 'docs/v1/06_WEB_PLAN.md')) {
	const data = join(engineRoot, 'data');
	const genDir = join(data, 'generations');
	const index = readJson(join(genDir, 'index.json'));
	for (const g of index.generations) need(join(genDir, g.file));
	for (const f of Object.values(PRESET_FILES)) need(join(data, f));

	// Champion lineage: sum training.games along the parent chain, cross-checked
	// against the results files that record each self-play run.
	const gens = new Map(index.generations.map((g) => [g.generation, readJson(join(genDir, g.file))]));
	const champion = readJson(join(data, CHAMPION_FILE));
	const lineage = [];
	let cur = champion;
	while (cur) {
		const n = cur.generation;
		const fromFile = cur.training?.games ?? 0;
		let fromResults = null;
		if (LINEAGE_RESULTS[n]) {
			fromResults = readJson(join(engineRoot, LINEAGE_RESULTS[n])).training?.games;
			if (fromResults !== fromFile) {
				throw new Error(`sync-data: generation ${n} games differ: ${fromFile} (generation file) vs ${fromResults} (results)`);
			}
		}
		lineage.push({ generation: n, games: fromFile, source: LINEAGE_RESULTS[n] ?? null });
		cur = cur.parent == null ? null : gens.get(cur.parent);
		if (cur === undefined) throw new Error('sync-data: broken parent chain');
	}
	for (const [n, p] of Object.entries(LINEAGE_RESULTS)) {
		need(join(engineRoot, p));
		if (!lineage.some((l) => l.generation === Number(n))) throw new Error(`sync-data: generation ${n} not in champion lineage`);
	}
	const championGames = lineage.reduce((s, l) => s + l.games, 0);

	// Preset labels and levels come from the web plan tables, so the copy is never retyped.
	const plan = readFileSync(need(planPath), 'utf8');
	const presetRows = tableAfter(plan, '### 5c. Universe presets');
	const levelRows = tableAfter(plan, '### 5d. Levels');
	const championConstants = normConstants(champion.constants);

	const presets = Object.entries(PRESET_FILES).map(([slug, file]) => {
		const row = presetRows.find((r) => stripTicks(r[0]) === slug);
		if (!row) throw new Error(`sync-data: preset ${slug} missing from web plan 5c`);
		const doc = readJson(join(data, file));
		const constants = normConstants(doc.constants);
		const own = doc.training?.games ?? 0;
		// Untrained variants (parent == own generation) share the champion's lineage.
		const isVariant = doc.parent === doc.generation;
		const selfPlayGames = isVariant ? championGames : slug === 'champion' ? championGames : own;
		return {
			slug,
			name: row[1],
			label: stripQuotes(row[3]),
			file: `/data/${file}`,
			generation: doc.generation,
			selfPlayGames,
			constants,
			diff: constantsDiff(championConstants, constants)
		};
	});

	const levels = levelRows.map((r) => ({
		id: r[0].toLowerCase(),
		name: r[0],
		nodes: Number(r[1].replace(/,/g, '')),
		about: r[2]
	}));
	if (levels.length !== 3 || levels.some((l) => !Number.isFinite(l.nodes))) throw new Error('sync-data: could not read levels from web plan 5d');

	return {
		championId: index.champion,
		selfPlayGames: championGames,
		lineage,
		presets,
		levels,
		landing: computeLanding(engineRoot),
		defaultLevel: 'default'
	};
}

export function syncData(outDir = join(WEB_ROOT, 'static/data')) {
	const engineRoot = join(REPO_ROOT, 'engine');
	const facts = computeFacts(engineRoot);
	const data = join(engineRoot, 'data');
	rmSync(outDir, { recursive: true, force: true });
	const copy = (rel) => {
		const dest = join(outDir, rel);
		mkdirSync(dirname(dest), { recursive: true });
		copyFileSync(need(join(data, rel)), dest);
	};
	copy('generations/index.json');
	const index = readJson(join(data, 'generations/index.json'));
	for (const g of index.generations) copy(`generations/${g.file}`);
	for (const f of Object.values(PRESET_FILES)) copy(f);
	writeFileSync(join(outDir, 'site-facts.json'), JSON.stringify(facts, null, 1) + '\n');
	return facts;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	try {
		const f = syncData();
		console.log(`sync-data: ${f.presets.length} presets, champion lineage ${f.selfPlayGames} self-play games`);
	} catch (e) {
		console.error(String(e.message ?? e));
		process.exit(1);
	}
}
