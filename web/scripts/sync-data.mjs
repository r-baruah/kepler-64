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
	{ id: 'field', verdict: 'kept', title: 'Pull follows the moves', note: 'Beats counting pieces alone, by far', universes: [], evidence: ['results/ledger-gen001', 'results/ledger-gen001.md'] },
	{ id: 'space', verdict: 'kept', title: 'Squares belong to the stronger pull', note: 'Control of space', universes: [], evidence: ['results/ledger-gen001.md'] },
	{ id: 'binding', verdict: 'back', title: 'Defenders hold pieces together', note: 'Cut at first, useful later', universes: [], evidence: ['results/ledger-gen003-bind0.json'] },
	{ id: 'accretion', verdict: 'rejected', title: 'Captures absorb weight', note: 'Better switched off', universes: ['accretion'], evidence: ['results/ledger-gen001.md'] },
	{ id: 'king', verdict: 'rejected-twice', title: 'A king torn apart by pull', note: 'Two designs, neither helped', universes: ['strong-roche', 'tidal-load'], evidence: ['results/ledger-gen003-roche0.json', 'results/ledger-gen004tide-tide0.json'] }
];

/** The proof line on the landing: Gen 3 against Stockfish 19 at its 2100 setting, 30+0.3. */
const PROOF_FILE = 'results/ladder-gen003.json';
const PROOF_RUNG = 2100;

function computeProof(engineRoot) {
	const doc = readJson(join(engineRoot, PROOF_FILE));
	if (!/gen-003\.json$/.test(doc.gen ?? '')) throw new Error(`sync-data: ${PROOF_FILE} is not a Gen 3 ladder`);
	if (!/^sf_19\b/.test(doc.stockfish ?? '')) throw new Error(`sync-data: ${PROOF_FILE} is not against Stockfish 19`);
	const m = /^(\d+)\+([\d.]+)$/.exec(doc.tc ?? '');
	if (!m) throw new Error(`sync-data: ${PROOF_FILE} has no time control`);
	const rung = (doc.rungs ?? []).find((r) => r.stockfish_elo === PROOF_RUNG);
	if (!rung) throw new Error(`sync-data: ${PROOF_FILE} has no ${PROOF_RUNG} rung`);
	if (rung.w + rung.d + rung.l !== rung.games) throw new Error(`sync-data: ${PROOF_FILE} rung ${PROOF_RUNG} does not add up`);
	const score = (rung.w + rung.d / 2) / rung.games;
	if (Math.abs(score - rung.score) > 1e-9) throw new Error(`sync-data: ${PROOF_FILE} score mismatch`);
	return {
		generation: 3,
		opponent: 'Stockfish 19',
		setting: PROOF_RUNG,
		baseSeconds: Number(m[1]),
		incrementSeconds: Number(m[2]),
		games: rung.games,
		w: rung.w,
		d: rung.d,
		l: rung.l,
		scorePercent: Math.round(score * 100),
		source: PROOF_FILE
	};
}

function computeLanding(engineRoot) {
	const cards = LEDGER_CARDS.map((c) => {
		for (const e of c.evidence) need(join(engineRoot, e));
		for (const u of c.universes) if (!(u in PRESET_FILES)) throw new Error(`sync-data: ledger card ${c.id} names unknown universe ${u}`);
		return { ...c };
	});
	return { proof: computeProof(engineRoot), ledger: cards };
}

/**
 * BibTeX for the Cite button, built from CITATION.cff so the two never drift. Only title, authors
 * and url are used: no version, no DOI, no year (the repository is cited as it stands).
 * CITATION.cff is simple YAML; the few fields needed are read with patterns, no YAML dependency.
 */
export function bibtexFromCff(cffText) {
	const field = (re, what) => {
		const m = re.exec(cffText);
		if (!m) throw new Error(`sync-data: CITATION.cff has no ${what}`);
		return m[1].trim().replace(/^"(.*)"$/, '$1');
	};
	const title = field(/^title:\s*(.+)$/m, 'title');
	const url = field(/^url:\s*(.+)$/m, 'url');
	const authorsBlock = /^authors:\n((?:[ \t]+.*\n?)+)/m.exec(cffText);
	if (!authorsBlock) throw new Error('sync-data: CITATION.cff has no authors');
	const authors = authorsBlock[1]
		.split(/\n\s*-\s+/)
		.map((chunk) => {
			const fam = /family-names:\s*"?([^"\n]+)"?/.exec(chunk)?.[1]?.trim();
			const giv = /given-names:\s*"?([^"\n]+)"?/.exec(chunk)?.[1]?.trim();
			return fam ? (giv ? `${fam}, ${giv}` : fam) : null;
		})
		.filter(Boolean);
	if (!authors.length) throw new Error('sync-data: CITATION.cff authors have no family-names');
	const key = (authors[0].split(',')[0] + '-kepler64').toLowerCase().replace(/[^a-z0-9-]/g, '');
	const bib = `@software{${key},\n  title  = {${title}},\n  author = {${authors.join(' and ')}},\n  url    = {${url}}\n}`;
	return { title, authors, url, bibtex: bib };
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

export function computeFacts(
	engineRoot = join(REPO_ROOT, 'engine'),
	planPath = join(REPO_ROOT, 'docs/v1/06_WEB_PLAN.md'),
	cffPath = join(REPO_ROOT, 'CITATION.cff')
) {
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
			/** Home's constants with one law changed, never trained itself: the site shows no game count for it. */
			variant: isVariant && slug !== 'champion',
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
		cite: bibtexFromCff(readFileSync(need(cffPath), 'utf8')),
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
