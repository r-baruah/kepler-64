/* tslint:disable */
/* eslint-disable */
export class Engine {
  free(): void;
  legalMoves(): string[];
  /**
   * Merge a partial constants object (`{"rho": 2.0}`) into the current constants.
   */
  setConstants(partial_json: string): void;
  /**
   * The current constants as a JSON string.
   */
  constantsJson(): string;
  /**
   * `generationJson` is a generation file (`gen-XXX.json`) or a bare constants object.
   */
  constructor(generation_json: string);
  /**
   * Play a UCI move. Returns false if it is illegal or the game is over.
   */
  play(uci: string): boolean;
  /**
   * `{ fen, turn, result, reason, inCheck, ply, halfmoveClock }`
   */
  state(): any;
  /**
   * `limits`: `{ depth?, movetimeMs?, nodes?, nps? }` (all optional).
   */
  search(limits: any): any;
  /**
   * Reset to the start position, or to `fen` (accreted mass starts at zero).
   */
  newGame(fen?: string | null): void;
  /**
   * The `02_LAWS_SPEC.md` §10 breakdown as a JS object (see `eval::Report`).
   */
  breakdown(): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_engine_free: (a: number, b: number) => void;
  readonly engine_breakdown: (a: number) => [number, number, number];
  readonly engine_constantsJson: (a: number) => [number, number];
  readonly engine_legalMoves: (a: number) => [number, number];
  readonly engine_new: (a: number, b: number) => [number, number, number];
  readonly engine_newGame: (a: number, b: number, c: number) => [number, number];
  readonly engine_play: (a: number, b: number, c: number) => number;
  readonly engine_search: (a: number, b: any) => [number, number, number];
  readonly engine_setConstants: (a: number, b: number, c: number) => [number, number];
  readonly engine_state: (a: number) => [number, number, number];
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __externref_drop_slice: (a: number, b: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
