// PGN export: a pure function from the game's move list to standard PGN text.
import { fenTurn, START_FEN } from './chess';
import type { Side } from './types';

export interface PgnInput {
	/** SAN of each move in order, check and mate suffixes included. */
	sans: string[];
	/** The position the game began from; null or the standard start for a normal game. */
	startFen?: string | null;
	human: Side;
	result: '1-0' | '0-1' | '1/2-1/2' | null;
	universe: string;
	level: string;
	/** Local date of the export. */
	date: Date;
	/** Engine commit stamp; omitted from the tags when empty. */
	engineCommit?: string | null;
}

const pad = (n: number, w = 2) => String(n).padStart(w, '0');

/** YYYY.MM.DD in local time (the PGN date format). */
export function pgnDate(d: Date): string {
	return `${pad(d.getFullYear(), 4)}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}`;
}

/** YYYY-MM-DD in local time, for file names. */
export function fileDate(d: Date): string {
	return pgnDate(d).replaceAll('.', '-');
}

export const pgnFileName = (d: Date) => `kepler-64-${fileDate(d)}.pgn`;

const tag = (name: string, value: string) => `[${name} "${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"]`;

function wrap(tokens: string[], width = 80): string {
	const lines: string[] = [];
	let cur = '';
	for (const t of tokens) {
		if (cur && cur.length + 1 + t.length > width) {
			lines.push(cur);
			cur = t;
		} else {
			cur = cur ? `${cur} ${t}` : t;
		}
	}
	if (cur) lines.push(cur);
	return lines.join('\n');
}

export function buildPgn(g: PgnInput): string {
	const custom = !!g.startFen && g.startFen !== START_FEN;
	const result = g.result ?? '*';
	const tags = [
		tag('Event', 'Kepler-64 casual game'),
		tag('Site', 'https://kepler.ripuranjan.com'),
		tag('Date', pgnDate(g.date)),
		tag('Round', '-'),
		tag('White', g.human === 'white' ? 'You' : 'Kepler-64'),
		tag('Black', g.human === 'black' ? 'You' : 'Kepler-64'),
		tag('Result', result),
		tag('Universe', g.universe),
		tag('Level', g.level)
	];
	const commit = g.engineCommit?.trim();
	if (commit) tags.push(tag('EngineCommit', commit));
	if (custom) tags.push(tag('SetUp', '1'), tag('FEN', g.startFen!));

	// Move numbers continue from the start position's own counter.
	const startFen = custom ? g.startFen! : START_FEN;
	let blackToMove = fenTurn(startFen) === 'b';
	let num = Number(startFen.split(' ')[5]) || 1;
	const tokens: string[] = [];
	g.sans.forEach((san, i) => {
		if (!blackToMove) tokens.push(`${num}.`);
		else if (i === 0) tokens.push(`${num}...`);
		tokens.push(san);
		if (blackToMove) num++;
		blackToMove = !blackToMove;
	});
	tokens.push(result);
	return `${tags.join('\n')}\n\n${wrap(tokens)}\n`;
}
