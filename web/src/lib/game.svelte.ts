// The Play screen's state: one game against Kepler, driven through the engine worker.
// All chess and evaluation work happens in the engine; this file only sequences calls.
import { commentary } from './commentary';
import { fenTurn, moveNumber, parsePlacement, sanFor, sqIndex, sqName, START_FEN, withSuffix, type Occupant } from './chess';
import { EngineClient, webAssemblySupported } from './engine/client';
import engineCommit from './engine/pkg/ENGINE_COMMIT?raw';
import { buildPgn, pgnFileName } from './pgn';
import type { EngineState, LevelFacts, PresetFacts, Side, SiteFacts, Snapshot } from './types';

export type FieldMode = 'off' | 'pull' | 'cones';
export type Sheet = null | 'universe' | 'level' | 'new' | 'more';

export interface HistoryItem {
	uci: string;
	san: string;
	mover: 'you' | 'kepler';
	/** Kepler's search score for the position it moved from, in ku, from its own side; null for your moves. */
	scoreKu: number | null;
	mateIn: number | null;
}

/** Nodes for "Move now": a very small search so the reply is immediate. */
const MOVE_NOW_NODES = 3000;

export const FIELD_ORDER: FieldMode[] = ['pull', 'cones', 'off'];

export function resultText(s: EngineState, human: Side): string {
	if (!s.result) return '';
	const humanWon = (s.result === '1-0') === (human === 'white');
	switch (s.reason) {
		case 'checkmate':
			return humanWon ? 'Checkmate. You won.' : 'Checkmate. Kepler won.';
		case 'stalemate':
			return 'Draw by stalemate.';
		case 'threefold_repetition':
			return 'Draw by repetition.';
		case 'fifty_move_rule':
			return 'Draw by the fifty-move rule.';
		case 'insufficient_material':
			return 'Draw: not enough pieces left to mate.';
		default:
			return 'Game over.';
	}
}

export class GameStore {
	facts: SiteFacts;
	client = new EngineClient();

	presetSlug = $state('champion');
	levelId = $state<LevelFacts['id']>('default');
	mode = $state<'rest' | 'play'>('rest');
	human = $state<Side>('white');
	flipped = $state(false);
	fieldMode = $state<FieldMode>('pull');
	status = $state<'loading' | 'ready' | 'nowasm' | 'error'>('loading');
	sheet = $state<Sheet>(null);

	snap = $state.raw<Snapshot | null>(null);
	history = $state.raw<HistoryItem[]>([]);
	thinking = $state(false);
	thinkStart = $state(0);
	line = $state('');
	lineMeta = $state('');
	lastMove = $state.raw<{ from: number; to: number } | null>(null);
	selected = $state<number | null>(null);
	coneSq = $state<number | null>(null);
	promotion = $state.raw<{ from: string; to: string } | null>(null);

	private epoch = 0;
	private busy = $state(false);
	private startFen: string | null = null;
	private genCache = new Map<string, string>();

	constructor(facts: SiteFacts) {
		this.facts = facts;
	}

	// ---- derived views -------------------------------------------------------------

	get preset(): PresetFacts {
		return this.facts.presets.find((p) => p.slug === this.presetSlug) ?? this.facts.presets[0];
	}
	get level(): LevelFacts {
		return this.facts.levels.find((l) => l.id === this.levelId) ?? this.facts.levels[1];
	}
	get fen(): string {
		return this.snap?.state.fen ?? this.startFen ?? START_FEN;
	}
	get placement(): (Occupant | null)[] {
		return parsePlacement(this.fen);
	}
	get over(): boolean {
		return !!this.snap?.state.result;
	}
	get turnSide(): Side {
		return (this.snap ? this.snap.state.turn : fenTurn(this.fen)) === 'w' ? 'white' : 'black';
	}
	get myTurn(): boolean {
		return this.mode === 'play' && this.status === 'ready' && !this.over && !this.thinking && !this.busy && this.turnSide === this.human;
	}
	get keplerSide(): Side {
		return this.human === 'white' ? 'black' : 'white';
	}
	/** The side drawn at the bottom of the board. */
	get bottom(): Side {
		return (this.human === 'white') !== this.flipped ? 'white' : 'black';
	}
	/** Legal destinations of the selected piece. */
	get targets(): Map<number, 'move' | 'capture'> {
		const out = new Map<number, 'move' | 'capture'>();
		if (this.selected === null || !this.snap || !this.myTurn) return out;
		const from = sqName(this.selected);
		const board = this.placement;
		for (const m of this.snap.legal) {
			if (!m.startsWith(from)) continue;
			const to = sqIndex(m.slice(2, 4));
			const pawnDiag = board[this.selected]?.piece === 'P' && to % 8 !== this.selected % 8;
			out.set(to, board[to] || pawnDiag ? 'capture' : 'move');
		}
		return out;
	}
	/** Hop-1 and hop-2 squares of the cone piece, when the Light cones layer is on. */
	get cones(): { origin: number; hop1: number[]; hop2: number[] } | null {
		if (this.fieldMode !== 'cones' || this.coneSq === null || !this.snap) return null;
		const name = sqName(this.coneSq);
		const p = this.snap.breakdown.pieces.find((x) => x.square === name);
		if (!p) return null;
		return { origin: this.coneSq, hop1: p.hops[0].map(sqIndex), hop2: p.hops[1].map(sqIndex) };
	}
	/** The moved piece's hop-1 squares after the last move (its pull lines fade in once, Pull layer only). */
	get pulse(): { key: string; origin: number; hop1: number[] } | null {
		if (this.fieldMode !== 'pull' || !this.lastMove || !this.snap || this.history.length === 0) return null;
		const name = sqName(this.lastMove.to);
		const p = this.snap.breakdown.pieces.find((x) => x.square === name);
		if (!p) return null;
		return { key: `${this.history.length}:${name}`, origin: this.lastMove.to, hop1: p.hops[0].map(sqIndex) };
	}
	/** Square of the king in check, if the side to move is in check. */
	get checkSquare(): number | null {
		if (!this.snap?.state.inCheck) return null;
		const k = this.snap.breakdown.kings[this.snap.state.turn === 'w' ? 0 : 1];
		return k ? sqIndex(k.square) : null;
	}
	/** Kepler's view in ku, from Kepler's side: its search score from its last move (the same
	 *  number the move list shows), or the static evaluation before Kepler has moved. */
	get keplerView(): number | null {
		for (let i = this.history.length - 1; i >= 0; i--) {
			const h = this.history[i];
			if (h.mover === 'kepler' && h.scoreKu !== null) return h.scoreKu;
		}
		if (!this.snap) return null;
		const w = this.snap.breakdown.terms.score_white;
		return this.keplerSide === 'white' ? w : -w;
	}
	king(side: Side) {
		return this.snap?.breakdown.kings[side === 'white' ? 0 : 1] ?? null;
	}

	// ---- lifecycle -----------------------------------------------------------------

	async boot(startFen: string | null) {
		if (!webAssemblySupported()) {
			this.status = 'nowasm';
			return;
		}
		this.startFen = startFen;
		await this.loadPreset();
	}

	private async generationText(p: PresetFacts): Promise<string> {
		let t = this.genCache.get(p.slug);
		if (!t) {
			const res = await fetch(p.file);
			if (!res.ok) throw new Error(`could not load ${p.file}`);
			t = await res.text();
			this.genCache.set(p.slug, t);
		}
		return t;
	}

	private clearGameState() {
		this.history = [];
		this.thinking = false;
		this.line = '';
		this.lineMeta = '';
		this.lastMove = null;
		this.selected = null;
		this.coneSq = null;
		this.promotion = null;
		this.busy = false;
	}

	/** (Re)start the engine on the current preset: always a fresh game at the start position. */
	private async loadPreset() {
		const ep = ++this.epoch;
		this.status = 'loading';
		this.clearGameState();
		try {
			const text = await this.generationText(this.preset);
			let snap = await this.client.start(text);
			if (ep !== this.epoch) return;
			if (this.startFen) snap = await this.client.reset(this.startFen, []);
			if (ep !== this.epoch) return;
			this.snap = snap;
			this.status = 'ready';
		} catch (e) {
			if (ep !== this.epoch) return;
			console.error(e);
			this.status = 'error';
		}
	}

	/** Begin a game as `color`: turns the landing hero into the Play layout. */
	async start(color: Side) {
		this.human = color;
		this.flipped = false;
		this.mode = 'play';
		this.sheet = null;
		if (this.status === 'loading') {
			// The board is already up; the first move waits for the engine to wake.
			await this.waitReady();
		}
		if (this.status !== 'ready') return;
		const ep = ++this.epoch;
		this.clearGameState();
		this.snap = await this.client.reset(this.startFen, []);
		if (ep !== this.epoch) return;
		if (this.turnSide !== this.human) await this.keplerMove(ep);
	}

	/** True while a game has moves and is not finished. */
	get inProgress(): boolean {
		return this.mode === 'play' && this.history.length > 0 && !this.over;
	}

	/** Back to the landing (Play at rest) on a fresh board. Asks first if a game is in progress. */
	async goHome() {
		if (this.mode !== 'play') return;
		if (this.inProgress && typeof confirm === 'function' && !confirm('Leave this game and go back to the start page?')) return;
		this.mode = 'rest';
		this.sheet = null;
		if (this.status === 'nowasm') return;
		await this.loadPreset();
	}

	private async waitReady() {
		while (this.status === 'loading') await new Promise((r) => setTimeout(r, 50));
	}

	async setPreset(slug: string) {
		this.sheet = null;
		if (slug === this.presetSlug) return;
		this.presetSlug = slug;
		await this.loadPreset();
		if (this.mode === 'play' && this.status === 'ready') await this.start(this.human);
	}

	setLevel(id: LevelFacts['id']) {
		this.levelId = id;
		this.sheet = null;
	}

	cycleField() {
		const i = FIELD_ORDER.indexOf(this.fieldMode);
		this.fieldMode = FIELD_ORDER[(i + 1) % FIELD_ORDER.length];
		if (this.fieldMode !== 'cones') this.coneSq = null;
	}

	setField(mode: FieldMode) {
		this.fieldMode = mode;
		if (mode !== 'cones') this.coneSq = null;
	}

	flip() {
		this.flipped = !this.flipped;
		this.sheet = null;
	}

	// ---- board input ---------------------------------------------------------------

	/** Pointer went down on `sq`. Returns true if it picked up one of your pieces. */
	selectOwn(sq: number): boolean {
		const occ = this.placement[sq];
		if (!occ || !this.myTurn || occ.color !== this.human) return false;
		this.selected = sq;
		if (this.fieldMode === 'cones') this.coneSq = sq;
		return true;
	}

	/** A tap that did not pick up a piece. */
	tap(sq: number) {
		if (this.selected !== null && this.targets.has(sq)) {
			void this.tryMove(this.selected, sq);
			return;
		}
		const occ = this.placement[sq];
		if (this.fieldMode === 'cones' && occ) {
			this.coneSq = sq;
			this.selected = null;
			return;
		}
		this.selected = null;
		if (this.fieldMode === 'cones') this.coneSq = null;
	}

	async tryMove(from: number, to: number) {
		if (!this.snap || !this.myTurn) return;
		const f = sqName(from);
		const t = sqName(to);
		if (this.snap.legal.includes(f + t + 'q')) {
			this.promotion = { from: f, to: t };
			return;
		}
		if (!this.snap.legal.includes(f + t)) {
			this.selected = null;
			return;
		}
		await this.humanMove(f + t);
	}

	async choosePromotion(letter: 'q' | 'r' | 'b' | 'n') {
		const p = this.promotion;
		this.promotion = null;
		if (p) await this.humanMove(p.from + p.to + letter);
	}

	private async humanMove(uci: string) {
		const before = this.snap;
		if (!before || !this.myTurn) return;
		const ep = this.epoch;
		this.busy = true;
		this.selected = null;
		this.coneSq = null;
		try {
			const san = sanFor(before.state.fen, uci, before.legal);
			const r = await this.client.play(uci);
			if (ep !== this.epoch || !r.legal) return;
			this.applyMove(uci, 'you', before, r.snapshot, san, null, null);
		} catch (e) {
			if (ep === this.epoch) console.error(e);
			return;
		} finally {
			if (ep === this.epoch) this.busy = false;
		}
		if (ep === this.epoch && !this.over) await this.keplerMove(ep);
	}

	private applyMove(
		uci: string,
		mover: 'you' | 'kepler',
		before: Snapshot,
		after: Snapshot,
		san: string,
		scoreKu: number | null,
		mateIn: number | null
	) {
		const full = withSuffix(san, after.state.inCheck, after.state.reason === 'checkmate');
		const ply = this.history.length + (this.startFen && fenTurn(this.startFen) === 'b' ? 1 : 0);
		this.history = [...this.history, { uci, san: full, mover, scoreKu, mateIn }];
		this.snap = after;
		this.lastMove = { from: sqIndex(uci.slice(0, 2)), to: sqIndex(uci.slice(2, 4)) };
		this.line = commentary(before.breakdown, after.breakdown, { uci, mover });
		this.lineMeta = `${moveNumber(ply)} ${full}`;
	}

	private async keplerMove(ep: number, nodes = this.level.nodes) {
		const before = this.snap;
		if (!before || ep !== this.epoch) return;
		this.thinking = true;
		this.thinkStart = performance.now();
		try {
			const out = await this.client.search(nodes);
			if (ep !== this.epoch) return;
			const san = sanFor(before.state.fen, out.best, before.legal);
			const r = await this.client.play(out.best);
			if (ep !== this.epoch || !r.legal) return;
			this.thinking = false;
			this.applyMove(out.best, 'kepler', before, r.snapshot, san, out.scoreKu, out.mateIn);
		} catch (e) {
			if (ep === this.epoch) {
				console.error(e);
				this.thinking = false;
				this.status = 'error';
			}
		}
	}

	/** Cancel the search (by restarting the worker) and play a quick move instead. */
	async moveNow() {
		if (!this.thinking) return;
		const ep = ++this.epoch;
		try {
			await this.client.restart();
			this.snap = await this.client.reset(this.startFen, this.history.map((h) => h.uci));
			if (ep !== this.epoch) return;
			await this.keplerMove(ep, MOVE_NOW_NODES);
		} catch (e) {
			if (ep === this.epoch) {
				console.error(e);
				this.status = 'error';
			}
		}
	}

	get canUndo(): boolean {
		return this.mode === 'play' && this.status === 'ready' && !this.busy && this.history.some((h) => h.mover === 'you');
	}

	/** Take back your last move and everything after it. */
	async undo() {
		if (!this.canUndo) return;
		const items = [...this.history];
		while (items.length) {
			const it = items.pop()!;
			if (it.mover === 'you') break;
		}
		const wasThinking = this.thinking;
		const ep = ++this.epoch;
		this.thinking = false;
		this.history = items;
		this.line = '';
		this.lineMeta = '';
		this.selected = null;
		this.coneSq = null;
		this.promotion = null;
		const last = items[items.length - 1];
		this.lastMove = last ? { from: sqIndex(last.uci.slice(0, 2)), to: sqIndex(last.uci.slice(2, 4)) } : null;
		try {
			if (wasThinking) await this.client.restart();
			const snap = await this.client.reset(this.startFen, items.map((h) => h.uci));
			if (ep !== this.epoch) return;
			this.snap = snap;
		} catch (e) {
			if (ep === this.epoch) {
				console.error(e);
				this.status = 'error';
			}
		}
	}

	// ---- PGN export ----------------------------------------------------------------

	pgnCopied = $state(false);
	private copiedTimer: ReturnType<typeof setTimeout> | undefined;

	/** PGN of the game so far (result "*" while unfinished). */
	pgn(date = new Date()): string {
		return buildPgn({
			sans: this.history.map((h) => h.san),
			startFen: this.startFen,
			human: this.human,
			result: this.snap?.state.result ?? null,
			universe: this.preset.name,
			level: this.level.name,
			date,
			engineCommit
		});
	}

	async copyPgn() {
		try {
			await navigator.clipboard.writeText(this.pgn());
		} catch (e) {
			console.error(e);
			return;
		}
		this.pgnCopied = true;
		clearTimeout(this.copiedTimer);
		this.copiedTimer = setTimeout(() => (this.pgnCopied = false), 1500);
	}

	downloadPgn() {
		const date = new Date();
		const url = URL.createObjectURL(new Blob([this.pgn(date)], { type: 'application/x-chess-pgn' }));
		const a = document.createElement('a');
		a.href = url;
		a.download = pgnFileName(date);
		document.body.appendChild(a);
		a.click();
		a.remove();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}

	dispose() {
		clearTimeout(this.copiedTimer);
		this.epoch++;
		this.client.stop();
	}
}
