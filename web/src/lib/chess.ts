// Small, pure chess helpers: squares, FEN placement, SAN. No evaluation knowledge lives here.
import type { PieceLetter, Side } from './types';

export const FILES = 'abcdefgh';
export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

/** Square name to index, a1 = 0, h8 = 63. */
export function sqIndex(name: string): number {
	return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
}

export function sqName(i: number): string {
	return FILES[i % 8] + String(Math.floor(i / 8) + 1);
}

export interface Occupant {
	piece: PieceLetter;
	color: Side;
}

/** Piece placement of a FEN, indexed a1 = 0. */
export function parsePlacement(fen: string): (Occupant | null)[] {
	const out: (Occupant | null)[] = new Array(64).fill(null);
	const rows = fen.split(' ')[0].split('/');
	for (let r = 0; r < 8; r++) {
		let f = 0;
		for (const ch of rows[r] ?? '') {
			if (ch >= '1' && ch <= '8') {
				f += Number(ch);
			} else {
				const rank = 7 - r;
				out[rank * 8 + f] = { piece: ch.toUpperCase() as PieceLetter, color: ch === ch.toUpperCase() ? 'white' : 'black' };
				f++;
			}
		}
	}
	return out;
}

export function fenTurn(fen: string): 'w' | 'b' {
	return fen.split(' ')[1] === 'b' ? 'b' : 'w';
}

/** Move number text for a ply index (0 = White's first move): "1." or "1...". */
export function moveNumber(ply: number): string {
	const n = Math.floor(ply / 2) + 1;
	return ply % 2 === 0 ? `${n}.` : `${n}…`;
}

/**
 * Standard algebraic notation for `uci`, played from `fen` with `legal` the legal UCI moves there.
 * The check or mate suffix is added by the caller (`withSuffix`) once the next state is known.
 */
export function sanFor(fen: string, uci: string, legal: string[]): string {
	const board = parsePlacement(fen);
	const from = sqIndex(uci.slice(0, 2));
	const to = sqIndex(uci.slice(2, 4));
	const promo = uci[4]?.toUpperCase();
	const mover = board[from];
	if (!mover) return uci;
	const target = board[to];
	if (mover.piece === 'K' && Math.abs((to % 8) - (from % 8)) === 2 && Math.floor(to / 8) === Math.floor(from / 8)) {
		return to % 8 > from % 8 ? 'O-O' : 'O-O-O';
	}
	const enPassant = mover.piece === 'P' && !target && (to % 8) !== (from % 8);
	const capture = !!target || enPassant;
	const dest = sqName(to);
	if (mover.piece === 'P') {
		let s = capture ? FILES[from % 8] + 'x' + dest : dest;
		if (promo) s += '=' + promo;
		return s;
	}
	// Disambiguation: other legal moves of the same piece type to the same square.
	const rivals = legal.filter((m) => {
		if (m === uci || m.slice(2, 4) !== uci.slice(2, 4)) return false;
		const o = board[sqIndex(m.slice(0, 2))];
		return o?.piece === mover.piece && o.color === mover.color;
	});
	let dis = '';
	if (rivals.length) {
		const sameFile = rivals.some((m) => m[0] === uci[0]);
		const sameRank = rivals.some((m) => m[1] === uci[1]);
		if (!sameFile) dis = uci[0];
		else if (!sameRank) dis = uci[1];
		else dis = uci.slice(0, 2);
	}
	return mover.piece + dis + (capture ? 'x' : '') + dest;
}

export function withSuffix(san: string, inCheck: boolean, checkmate: boolean): string {
	return checkmate ? san + '#' : inCheck ? san + '+' : san;
}
