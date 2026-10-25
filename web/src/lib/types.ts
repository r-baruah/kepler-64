// Shapes returned by the kepler-wasm Engine (docs/v1/03_ENGINE_ARCHITECTURE.md 6.2).
// Report fields are snake_case (they come straight from Rust structs); state and search are camelCase.

export type Side = 'white' | 'black';
export type PieceLetter = 'P' | 'N' | 'B' | 'R' | 'Q' | 'K';

export interface PieceReport {
	square: string;
	piece: PieceLetter;
	color: Side;
	rest_mass: number;
	accreted: number;
	carried_mass: number;
	emission: number;
	energy: number;
	/** Squares at hop distance 1, 2 and 3. */
	hops: [string[], string[], string[]];
}

export interface KingReport {
	square: string;
	body: string[];
	/** Enemy pull across the king's body. */
	stress: number;
	/** Pull from the king's own side across its body. */
	binding: number;
	eta: number;
	gauge: number;
	break_ratio: number;
}

export interface Constants {
	alpha: number;
	beta: number;
	c: number;
	f_acc: number;
	k_roche: number;
	kappa: number;
	lambda_bind: number;
	lambda_space: number;
	lambda_tide: number;
	m_b: number;
	m_n: number;
	m_q: number;
	m_r: number;
	rho: number;
	sigma: number;
}

export interface Breakdown {
	terms: { material: number; bind: number; space: number; roche: number; tide: number; score_white: number };
	/** 64 values each, a1 = 0, b1 = 1, ..., h8 = 63. */
	field: { white: number[]; black: number[] };
	/** White first, then Black. */
	pieces: PieceReport[];
	/** White's king, then Black's. */
	kings: [KingReport, KingReport];
	constants: Constants;
	generation: number | null;
}

export interface EngineState {
	fen: string;
	turn: 'w' | 'b';
	result: '1-0' | '0-1' | '1/2-1/2' | null;
	reason: 'checkmate' | 'stalemate' | 'fifty_move_rule' | 'insufficient_material' | 'threefold_repetition' | null;
	inCheck: boolean;
	ply: number;
	halfmoveClock: number;
}

export interface SearchOut {
	best: string;
	scoreKu: number;
	mateIn: number | null;
	depth: number;
	seldepth: number;
	nodes: number;
	pv: string[];
}

export interface Snapshot {
	state: EngineState;
	legal: string[];
	breakdown: Breakdown;
}

export interface MoveInfo {
	uci: string;
	mover: 'kepler' | 'you';
}

// Contents of static/data/site-facts.json (written by scripts/sync-data.mjs).
export interface PresetFacts {
	slug: string;
	name: string;
	label: string;
	file: string;
	generation: number;
	/** Home's constants with one law changed; never trained, so no game count is shown. */
	variant: boolean;
	selfPlayGames: number;
	constants: Constants;
	diff: Record<string, { champion: number; preset: number }>;
}

export interface LevelFacts {
	id: 'quick' | 'default' | 'strong';
	name: string;
	nodes: number;
	about: string;
}

export interface LedgerCard {
	id: string;
	verdict: 'kept' | 'back' | 'rejected' | 'rejected-twice';
	title: string;
	note: string;
	/** Preset slugs of the alternate universes this law lives on in (empty for kept laws). */
	universes: string[];
	evidence: string[];
}

/** The proof line: one ladder rung, read from engine/results by scripts/sync-data.mjs. */
export interface ProofFacts {
	generation: number;
	opponent: string;
	/** Stockfish's own strength setting (UCI_Elo), not a human rating. */
	setting: number;
	baseSeconds: number;
	incrementSeconds: number;
	games: number;
	w: number;
	d: number;
	l: number;
	scorePercent: number;
	source: string;
}

export interface LandingFacts {
	proof: ProofFacts;
	ledger: LedgerCard[];
}

/** Built from CITATION.cff by scripts/sync-data.mjs: title, authors and url only. */
export interface CiteFacts {
	title: string;
	authors: string[];
	url: string;
	bibtex: string;
}

export interface SiteFacts {
	landing: LandingFacts;
	cite: CiteFacts;
	championId: string;
	selfPlayGames: number;
	lineage: { generation: number; games: number; source: string | null }[];
	presets: PresetFacts[];
	levels: LevelFacts[];
	defaultLevel: string;
}
