// The engine lives here, off the main thread. Search is synchronous inside wasm, so this worker
// cannot be interrupted mid-search; the main thread cancels by terminating it (see client.ts).
import init, { Engine } from './pkg/kepler_wasm.js';
import type { Breakdown, EngineState, SearchOut } from '../types';
import type { Request, Response } from './protocol';

let engine: Engine | null = null;

function snapshot() {
	const e = engine!;
	return {
		state: e.state() as EngineState,
		legal: e.legalMoves(),
		breakdown: e.breakdown() as Breakdown
	};
}

async function handle(req: Request): Promise<unknown> {
	switch (req.type) {
		case 'init': {
			await init({ module_or_path: req.module ?? req.wasmUrl });
			engine?.free();
			engine = new Engine(req.generation);
			engine.newGame();
			return snapshot();
		}
		case 'reset': {
			const e = engine!;
			e.newGame(req.fen);
			for (const m of req.moves) {
				if (!e.play(m)) throw new Error(`illegal move in replay: ${m}`);
			}
			return snapshot();
		}
		case 'play': {
			const legal = engine!.play(req.uci);
			return { legal, snapshot: snapshot() };
		}
		case 'search': {
			// Only a node limit: the chosen move is then the same on every device.
			return engine!.search({ nodes: req.nodes }) as SearchOut;
		}
	}
}

self.onmessage = async (ev: MessageEvent<Request>) => {
	const req = ev.data;
	let res: Response;
	try {
		res = { id: req.id, ok: true, result: await handle(req) };
	} catch (e) {
		res = { id: req.id, ok: false, error: e instanceof Error ? e.message : String(e) };
	}
	(self as unknown as Worker).postMessage(res);
};
