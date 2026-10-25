import type { SearchOut, Snapshot } from '../types';

export type Request = { id: number } & (
	| { type: 'init'; module: WebAssembly.Module | null; wasmUrl: string; generation: string }
	| { type: 'reset'; fen: string | null; moves: string[] }
	| { type: 'play'; uci: string }
	| { type: 'search'; nodes: number }
);

export type Response = { id: number } & ({ ok: true; result: unknown } | { ok: false; error: string });

export type PlayResult = { legal: boolean; snapshot: Snapshot };
export type { SearchOut, Snapshot };
