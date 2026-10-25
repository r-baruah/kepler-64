// Main-thread handle on the engine worker. The wasm is compiled once here and the compiled module
// is posted to each worker, so restarting the worker (to cancel a search) is cheap.
import wasmUrl from './pkg/kepler_wasm_bg.wasm?url';
import type { PlayResult, Request, Response, SearchOut, Snapshot } from './protocol';

type Distribute<T> = T extends unknown ? Omit<T, 'id'> : never;
type Body = Distribute<Request>;

let modulePromise: Promise<WebAssembly.Module | null> | null = null;

/** Compile the engine once. Resolves to null if compiling here is not possible (the worker then fetches it). */
export function compileEngine(): Promise<WebAssembly.Module | null> {
	modulePromise ??= (async () => {
		try {
			const res = await fetch(wasmUrl);
			if (typeof WebAssembly.compileStreaming === 'function' && res.headers.get('content-type')?.includes('wasm')) {
				return await WebAssembly.compileStreaming(res);
			}
			return await WebAssembly.compile(await res.arrayBuffer());
		} catch {
			return null;
		}
	})();
	return modulePromise;
}

export function webAssemblySupported(): boolean {
	return typeof WebAssembly === 'object' && typeof Worker === 'function';
}

export class EngineClient {
	private worker: Worker | null = null;
	private nextId = 1;
	private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
	private generation = '';

	private spawn() {
		const w = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
		w.onmessage = (ev: MessageEvent<Response>) => {
			const p = this.pending.get(ev.data.id);
			if (!p) return;
			this.pending.delete(ev.data.id);
			if (ev.data.ok) p.resolve(ev.data.result);
			else p.reject(new Error(ev.data.error));
		};
		w.onerror = (ev) => {
			const err = new Error(ev.message || 'engine worker failed');
			for (const p of this.pending.values()) p.reject(err);
			this.pending.clear();
		};
		this.worker = w;
		return w;
	}

	private call<T>(body: Body): Promise<T> {
		const w = this.worker ?? this.spawn();
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
			try {
				w.postMessage({ ...body, id });
			} catch (e) {
				// A compiled module that cannot be cloned: let the worker fetch the wasm itself.
				if (body.type === 'init' && body.module) {
					w.postMessage({ ...body, module: null, id });
				} else {
					this.pending.delete(id);
					reject(e as Error);
				}
			}
		});
	}

	/** Start (or restart) a worker for a generation file's text. Resolves with the start position. */
	async start(generation: string): Promise<Snapshot> {
		this.generation = generation;
		this.stop();
		const module = await compileEngine();
		return this.call<Snapshot>({ type: 'init', module, wasmUrl, generation });
	}

	/** Terminate the worker (this is how a search is cancelled) and start a fresh one. */
	restart(): Promise<Snapshot> {
		return this.start(this.generation);
	}

	reset(fen: string | null, moves: string[]): Promise<Snapshot> {
		return this.call<Snapshot>({ type: 'reset', fen, moves });
	}

	play(uci: string): Promise<PlayResult> {
		return this.call<PlayResult>({ type: 'play', uci });
	}

	search(nodes: number): Promise<SearchOut> {
		return this.call<SearchOut>({ type: 'search', nodes });
	}

	stop() {
		this.worker?.terminate();
		this.worker = null;
		const err = new Error('engine worker stopped');
		for (const p of this.pending.values()) p.reject(err);
		this.pending.clear();
	}
}
