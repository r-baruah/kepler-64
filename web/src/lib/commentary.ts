// One line of commentary under the board. A pure function of two breakdowns and the move:
// it fills fixed templates with facts read from the engine's own report, and says nothing else.
//
// Template (one sentence):
//   "{Kepler's|Your} {piece} now pulls on {n} square(s)[ ({m} before)][, {clause}]."
// where the clause is at most one of
//   ", including {sq}, where it defends its {piece}"        (pieces that jump: knight, king, pawn)
//   ", blocked at {sq} by its own {piece}"                   (sliders: a ray whose last square holds its own piece)
//   ", including {sq}, where {your|Kepler's} {piece} stands" (hop-1 square holding an enemy piece)
// "n" is the number of squares at hop distance 1 of the moved piece after the move, "m" the same count
// for that piece before it moved.
import { FILES, sqIndex, sqName } from './chess';
import type { Breakdown, MoveInfo, PieceLetter, PieceReport } from './types';

const NAME: Record<PieceLetter, string> = { P: 'pawn', N: 'knight', B: 'bishop', R: 'rook', Q: 'queen', K: 'king' };

/** Ray directions as [file step, rank step] for each slider. */
const DIRS: Partial<Record<PieceLetter, [number, number][]>> = {
	R: [[1, 0], [-1, 0], [0, 1], [0, -1]],
	B: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
	Q: [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
};

const bySquare = (b: Breakdown) => new Map(b.pieces.map((p) => [p.square, p]));

/** Higher rest mass first (the king has none, so it sorts last), then lower square index. */
function heaviest(ps: PieceReport[]): PieceReport | undefined {
	return [...ps].sort((a, b) => b.rest_mass - a.rest_mass || sqIndex(a.square) - sqIndex(b.square))[0];
}

function blockedRay(piece: PieceReport, occ: Map<string, PieceReport>): PieceReport | undefined {
	const dirs = DIRS[piece.piece];
	if (!dirs) return undefined;
	const reach = new Set(piece.hops[0]);
	const f0 = FILES.indexOf(piece.square[0]);
	const r0 = Number(piece.square[1]) - 1;
	const blockers: PieceReport[] = [];
	for (const [df, dr] of dirs) {
		let f = f0 + df;
		let r = r0 + dr;
		let last: string | null = null;
		while (f >= 0 && f < 8 && r >= 0 && r < 8 && reach.has(sqName(r * 8 + f))) {
			last = sqName(r * 8 + f);
			f += df;
			r += dr;
		}
		// Only claim a block when the ray could have gone on: there is a square beyond it on the board.
		const beyondOnBoard = f >= 0 && f < 8 && r >= 0 && r < 8;
		const hit = last ? occ.get(last) : undefined;
		if (hit && hit.color === piece.color && beyondOnBoard) blockers.push(hit);
	}
	return heaviest(blockers);
}

export function commentary(before: Breakdown, after: Breakdown, move: MoveInfo): string {
	const from = move.uci.slice(0, 2);
	const to = move.uci.slice(2, 4);
	const promo = move.uci[4]?.toUpperCase() as PieceLetter | undefined;
	const was = before.pieces.find((p) => p.square === from);
	if (!was) return '';
	const now = after.pieces.find((p) => p.square === to && p.color === was.color);
	if (!now) return '';

	const owner = move.mover === 'kepler' ? "Kepler's" : 'Your';
	const enemyOwner = move.mover === 'kepler' ? 'your' : "Kepler's";
	const kind = promo ? `new ${NAME[promo]}` : NAME[now.piece];
	const n = now.hops[0].length;
	const squares = n === 0 ? 'no squares' : `${n} square${n === 1 ? '' : 's'}`;
	const prior = promo || was.piece === 'K' ? '' : was.hops[0].length !== n ? ` (${was.hops[0].length} before)` : '';

	const occ = bySquare(after);
	const inReach = now.hops[0].map((s) => occ.get(s)).filter((p): p is PieceReport => !!p);
	const own = inReach.filter((p) => p.color === now.color);
	const enemy = inReach.filter((p) => p.color !== now.color);

	let clause = '';
	const slider = !!DIRS[now.piece];
	if (slider) {
		const b = blockedRay(now, occ);
		if (b) clause = `, blocked at ${b.square} by its own ${NAME[b.piece]}`;
	} else {
		const d = heaviest(own);
		if (d) clause = `, including ${d.square}, where it defends its ${NAME[d.piece]}`;
	}
	if (!clause) {
		const king = enemy.find((p) => p.piece === 'K');
		const e = king ?? heaviest(enemy);
		if (e) clause = `, including ${e.square}, where ${enemyOwner} ${NAME[e.piece]} stands`;
	}
	return `${owner} ${kind} now pulls on ${squares}${prior}${clause}.`;
}
