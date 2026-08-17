/**
 * Kepler-64 Observatory Master Application Component
 * "forge"-inspired brutalist dark console — neon electric blue.
 */

import { Chess } from 'chess.js';
import { KeplerBoard } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { DEFAULT_CONSTANTS } from '../core/constants';
import { PRESET_GAMES } from '../core/presets';
import type { PresetGame } from '../core/presets';
import { DIST_64 } from '../core/gravity';
import { UnifiedCanvas } from '../render/UnifiedCanvas';
import type { ScoreBreakdown } from '../core/evaluate';
import { evaluatePosition } from '../core/evaluate';
import { ExportModal } from './ExportModal';
import { EvalSparkline } from './EvalSparkline';
import { FieldGuideComponent } from './FieldGuide';
import { ContributorsSection } from './ContributorsSection';
import { PgnImportModal } from './PgnImportModal';
import { latex } from './katexUtil';
import { BOT_PERSONAS, getPersona } from '../core/personas';
import type { SearchResult } from '../core/search';
import type { MultiverseSparkPoint } from './EvalSparkline';
import { buildAccretionLedger } from '../core/accretion';
import type { AccretionLedger } from '../core/accretion';

export class ObservatoryApp {
  private container: HTMLElement;
  private config: ConstantsConfig = { ...DEFAULT_CONSTANTS };
  private replayConfig: ConstantsConfig = { ...DEFAULT_CONSTANTS };
  private currentGame: PresetGame = PRESET_GAMES[0];
  private startFen: string | null = null;
  private chess: Chess = new Chess();
  private board: KeplerBoard = new KeplerBoard();
  private moves: any[] = [];
  private currentPlyIndex = 0;
  private isPlaying = false;
  private playIntervalId?: number;
  private mode: 'replay' | 'play' = 'replay';
  private playerColor: 'w' | 'b' = 'w';
  private personaId: string = BOT_PERSONAS[0].id;
  private liveChess: Chess = new Chess();
  private isBotThinking = false;
  private searchWorker: Worker | null = null;
  private multiverseWorker: Worker | null = null;
  private multiverseEnabled = false;
  private accretionExcess: Record<number, number> = {};
  private multiverseGen = 0;
  private searchGen = 0;
  private sparklineDebounce: number | undefined;
  private collapseCacheKey = '';
  private collapseCache: number[] = [];

  private canvasRenderer!: UnifiedCanvas;
  private exportModal!: ExportModal;
  private sparkline!: EvalSparkline;
  private fieldGuide!: FieldGuideComponent;
  private contributors!: ContributorsSection;
  private importModal!: PgnImportModal;

  constructor(container: HTMLElement) {
    this.container = container;
    this.initGame(this.currentGame);
    this.render();
    this.initCanvas();
    this.initSparkline();
    this.initFieldGuide();
    this.initContributors();
    this.exportModal = new ExportModal(this.currentGame, this.config);
    this.importModal = new PgnImportModal({
      onImportPgn: (pgn) => this.handleImportPgn(pgn),
      onImportFen: (fen) => this.handleImportFen(fen),
    });
  }

  private initGame(game: PresetGame): void {
    this.mode = 'replay';
    this.isBotThinking = false;
    if (this.sparklineDebounce) {
      window.clearTimeout(this.sparklineDebounce);
      this.sparklineDebounce = undefined;
    }
    this.cleanupWorker();
    this.cleanupMultiverseWorker();
    this.config = { ...this.replayConfig };
    this.syncConfigToUi();
    if (this.canvasRenderer) {
      this.canvasRenderer.setConfig(this.config);
      this.canvasRenderer.setOrientation('w');
    }
    this.currentGame = game;
    this.startFen =
      game.initialFen && game.initialFen !== 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        ? game.initialFen
        : null;
    try {
      this.chess = this.startFen ? new Chess(this.startFen) : new Chess();
    } catch {
      this.chess = new Chess();
    }
    if (game.pgn) this.chess.loadPgn(game.pgn);
    this.moves = this.chess.history({ verbose: true });
    this.currentPlyIndex = Math.max(0, Math.min(game.highlightPly, this.moves.length - 1));
    this.syncBoardToPly(this.currentPlyIndex);

    const slider = this.container?.querySelector('#ply-slider') as HTMLInputElement;
    if (slider) {
      slider.max = Math.max(0, this.moves.length - 1).toString();
      slider.value = this.currentPlyIndex.toString();
    }
    const plyIndicator = this.container?.querySelector('#ply-indicator');
    if (plyIndicator) {
      plyIndicator.textContent = `Ply ${this.currentPlyIndex + 1} / ${this.moves.length}`;
    }

    if (this.sparkline) {
      this.updateSparklineData();
    }
    this.updateModeDeckUI();
  }

  private syncBoardToPly(plyIndex: number): void {
    const tempChess = this.startFen ? new Chess(this.startFen) : new Chess();
    let lastMoveObj: { from: number; to: number } | null = null;

    for (let i = 0; i <= plyIndex; i++) {
      if (i < this.moves.length) {
        const m = this.moves[i];
        tempChess.move(m);
        if (i === plyIndex) {
          lastMoveObj = {
            from: (m.from.charCodeAt(1) - 49) * 8 + (m.from.charCodeAt(0) - 97),
            to: (m.to.charCodeAt(1) - 49) * 8 + (m.to.charCodeAt(0) - 97),
          };
        }
      }
    }

    this.board.loadFen(tempChess.fen());
    this.applyPhysicalBoosts(plyIndex, lastMoveObj);
    if (this.canvasRenderer) {
      this.canvasRenderer.setBoard(this.board, lastMoveObj);
      const inCheck = tempChess.inCheck();
      const checkSq = inCheck ? this.findKingSquare(tempChess.turn()) : null;
      this.canvasRenderer.setCheckSquare(checkSq);
    }
    if (this.sparkline) {
      this.sparkline.setCurrentPly(plyIndex);
    }
    this.updateCandidateMoves();
  }
  private initCanvas(): void {
    const canvas = this.container.querySelector('#board-canvas') as HTMLCanvasElement;
    if (!canvas) return;

    canvas.width = 600;
    canvas.height = 600;

    this.canvasRenderer = new UnifiedCanvas(canvas, this.board, this.config);

    this.canvasRenderer.onEvaluate((breakdown: ScoreBreakdown) => {
      this.updateTelemetry(breakdown);
    });

    this.canvasRenderer.onHover((info) => {
      const hud = this.container.querySelector('#hover-hud-text');
      if (hud) {
        if (info) {
          const pieceStr = info.piece ? `${info.piece.color === 'w' ? 'White' : 'Black'} ${info.piece.type.toUpperCase()} (${info.piece.mass.toFixed(1)}m)` : 'Empty Square';
          hud.textContent = `Square ${info.fileChar}${info.rankNum} · ${pieceStr} · Local Force: ${info.forceMag.toFixed(2)}N`;
        } else {
          hud.textContent = 'Hover over any square to inspect local gravitational force & mass';
        }
      }
    });
    this.canvasRenderer.onMove((fromSq, toSq) => {
      this.handleUserMove(fromSq, toSq);
    });

    this.canvasRenderer.setLegalMovesProvider((fromSq) => {
      const fromAlg = this.sqToAlg(fromSq);
      if (this.mode === 'play') {
        if (this.isBotThinking || this.liveChess.turn() !== this.playerColor) return [];
        const piece = this.board.squares[fromSq];
        if (!piece || piece.color !== this.playerColor) return [];
        const legal = this.liveChess.moves({ verbose: true });
        return legal.filter((m) => m.from === fromAlg).map((m) => this.algToSq(m.to));
      } else {
        try {
          const fen = this.board.toFen();
          const tempChess = new Chess(fen);
          const legal = tempChess.moves({ verbose: true });
          return legal.filter((m) => m.from === fromAlg).map((m) => this.algToSq(m.to));
        } catch {
          return [];
        }
      }
    });
    this.syncBoardToPly(this.currentPlyIndex);
  }

  private initSparkline(): void {
    const sparkContainer = this.container.querySelector('#sparkline-container') as HTMLElement;
    if (!sparkContainer) return;
    this.sparkline = new EvalSparkline(sparkContainer);
    this.sparkline.onSelectPly((ply) => {
      this.setPly(ply);
    });
    this.updateSparklineData();
  }

  private updateSparklineData(): void {
    const collapsePlies = this.computeCollapsePlies();
    this.sparkline.setCollapsePlies(collapsePlies);

    if (this.multiverseEnabled) {
      this.updateMultiverseSparkline();
      return;
    }
    this.cleanupMultiverseWorker();

    const tempChess = this.startFen ? new Chess(this.startFen) : new Chess();
    const tempBoard = new KeplerBoard();
    const points: { ply: number; score: number; moveSan: string }[] = [];

    for (let i = 0; i < this.moves.length; i++) {
      const m = this.moves[i];
      tempChess.move(m);
      tempBoard.loadFen(tempChess.fen());
      tempBoard.massBoost = this.computePlyBoost(
        this.moves.slice(0, i + 1),
        {
          from: (m.from.charCodeAt(1) - 49) * 8 + (m.from.charCodeAt(0) - 97),
          to: (m.to.charCodeAt(1) - 49) * 8 + (m.to.charCodeAt(0) - 97),
        },
        tempBoard
      );
      const evalRes = evaluatePosition(tempBoard, this.config);
      points.push({
        ply: i,
        score: evalRes.totalScoreWhite,
        moveSan: m.san,
      });
    }
    this.sparkline.setData(points, this.currentPlyIndex);
  }

  private scheduleSparklineRefresh(): void {
    if (this.sparklineDebounce) window.clearTimeout(this.sparklineDebounce);
    this.sparklineDebounce = window.setTimeout(() => this.updateSparklineData(), 120);
  }

  private computeCollapsePlies(): number[] {
    const key = `${this.startFen}|${this.moves.map((m) => m.san).join(' ')}|${JSON.stringify(this.config)}`;
    if (key === this.collapseCacheKey) return this.collapseCache;

    const plies: number[] = [];
    const tempChess = this.startFen ? new Chess(this.startFen) : new Chess();
    const tempBoard = new KeplerBoard();
    for (let i = 0; i < this.moves.length; i++) {
      const m = this.moves[i];
      tempChess.move(m);
      tempBoard.loadFen(tempChess.fen());
      tempBoard.massBoost = this.computePlyBoost(
        this.moves.slice(0, i + 1),
        {
          from: (m.from.charCodeAt(1) - 49) * 8 + (m.from.charCodeAt(0) - 97),
          to: (m.to.charCodeAt(1) - 49) * 8 + (m.to.charCodeAt(0) - 97),
        },
        tempBoard
      );
      const breakdown = evaluatePosition(tempBoard, this.config);
      if (breakdown.whiteKingTidal?.isDisrupted || breakdown.blackKingTidal?.isDisrupted) {
        plies.push(i);
      }
    }
    this.collapseCacheKey = key;
    this.collapseCache = plies;
    return plies;
  }

  private updateMultiverseSparkline(): void {
    this.multiverseGen++;
    const gen = this.multiverseGen;
    if (this.multiverseWorker) this.multiverseWorker.terminate();
    const worker = new Worker(new URL('../core/multiverseWorker.ts', import.meta.url), { type: 'module' });
    this.multiverseWorker = worker;

    worker.onmessage = (e: MessageEvent) => {
      if (gen !== this.multiverseGen || !this.multiverseEnabled) return;
      this.cleanupMultiverseWorker();
      const data = e.data as { ok?: boolean; points?: MultiverseSparkPoint[]; error?: string } | undefined;
      if (!data?.ok || !data.points) return;
      this.sparkline.setMultiverseData(data.points, this.currentPlyIndex);
    };

    worker.onerror = () => {
      if (gen !== this.multiverseGen) return;
      this.cleanupMultiverseWorker();
    };

    worker.postMessage({
      startFen: this.startFen ?? undefined,
      moves: this.moves.map((m) => m.san),
      config: this.config,
      sampleCount: 5,
      boosts: this.computeMultiverseBoosts(),
    });
  }

  private initFieldGuide(): void {
    const fgContainer = this.container.querySelector('#field-guide-mount') as HTMLElement;
    if (!fgContainer) return;
    this.fieldGuide = new FieldGuideComponent(fgContainer);
    this.fieldGuide.onSelectPosition((fen, _title) => {
      this.board.loadFen(fen);
      this.canvasRenderer.setBoard(this.board, null);
    });
    this.fieldGuide.render();
  }

  private initContributors(): void {
    const contContainer = this.container.querySelector('#contributors-mount') as HTMLElement;
    if (!contContainer) return;
    this.contributors = new ContributorsSection(contContainer);
    this.contributors.render();
  }

  private updateCandidateMoves(): void {
    const candidateList = this.container.querySelector('#candidate-moves-list');
    if (!candidateList) return;

    const tempChess = new Chess(this.board.toFen());
    const legalMoves = tempChess.moves({ verbose: true });
    if (!legalMoves.length) {
      candidateList.innerHTML = '<li style="color:var(--color-muted); font-size:0.8rem;">Terminal State</li>';
      return;
    }

    const moveEvaluations: { san: string; score: number }[] = [];
    const evalBoard = new KeplerBoard();

    const sampleMoves = legalMoves.slice(0, 4);
    sampleMoves.forEach((m) => {
      tempChess.move(m);
      evalBoard.loadFen(tempChess.fen());
      evalBoard.massBoost = this.computePlyBoost(
        [...this.moves.slice(0, this.currentPlyIndex + 1), m],
        {
          from: (m.from.charCodeAt(1) - 49) * 8 + (m.from.charCodeAt(0) - 97),
          to: (m.to.charCodeAt(1) - 49) * 8 + (m.to.charCodeAt(0) - 97),
        },
        evalBoard
      );
      const res = evaluatePosition(evalBoard, this.config);
      moveEvaluations.push({
        san: m.san,
        score: res.totalScoreMover,
      });
      tempChess.undo();
    });

    moveEvaluations.sort((a, b) => b.score - a.score);

    candidateList.innerHTML = moveEvaluations.map((m, idx) => `
      <li class="candidate-item">
        <div style="display:flex; align-items:center; gap:6px;">
          <span class="cand-rank">#${idx + 1}</span>
          <span class="cand-san">${m.san}</span>
        </div>
        <span class="cand-score">${(m.score >= 0 ? '+' : '') + m.score.toFixed(2)}</span>
      </li>
    `).join('');
  }

  private applyPhysicalBoosts(
    plyIndex: number,
    lastMoveObj: { from: number; to: number } | null
  ): void {
    const ledger = buildAccretionLedger(this.moves.slice(0, plyIndex + 1), this.config.accEta);
    this.accretionExcess = ledger.excessBySquare;

    const boost = new Float32Array(64);
    Object.entries(ledger.excessBySquare).forEach(([sqStr, excess]) => {
      boost[parseInt(sqStr, 10)] += excess;
    });

    // Relativistic Lorentz escalation for the most recent move.
    if (lastMoveObj) {
      const dist = DIST_64[lastMoveObj.from * 64 + lastMoveObj.to];
      const ratio = Math.min(0.95, dist / Math.max(0.1, this.config.c));
      const gamma = 1 / Math.sqrt(Math.max(0.05, 1 - ratio * ratio));
      const baseMass = this.board.squares[lastMoveObj.to]?.mass ?? 0;
      boost[lastMoveObj.to] += (gamma - 1) * (baseMass + (ledger.excessBySquare[lastMoveObj.to] ?? 0));
    }

    this.board.massBoost = boost;
    if (this.canvasRenderer) this.canvasRenderer.setAccretion(this.accretionExcess);
    this.updateAccretionHud(ledger);
  }

  private computePlyBoost(
    movesUpTo: any[],
    lastMoveObj: { from: number; to: number } | null,
    board: KeplerBoard
  ): Float32Array {
    const ledger = buildAccretionLedger(movesUpTo, this.config.accEta);

    const boost = new Float32Array(64);
    Object.entries(ledger.excessBySquare).forEach(([sqStr, excess]) => {
      boost[parseInt(sqStr, 10)] += excess;
    });

    // Relativistic Lorentz escalation for the most recent move.
    if (lastMoveObj) {
      const dist = DIST_64[lastMoveObj.from * 64 + lastMoveObj.to];
      const ratio = Math.min(0.95, dist / Math.max(0.1, this.config.c));
      const gamma = 1 / Math.sqrt(Math.max(0.05, 1 - ratio * ratio));
      const baseMass = board.squares[lastMoveObj.to]?.mass ?? 0;
      boost[lastMoveObj.to] += (gamma - 1) * (baseMass + (ledger.excessBySquare[lastMoveObj.to] ?? 0));
    }

    return boost;
  }

  private computeMultiverseBoosts(): Float32Array[] {
    const boosts: Float32Array[] = [];
    const tempChess = this.startFen ? new Chess(this.startFen) : new Chess();
    const tempBoard = new KeplerBoard();

    for (let i = 0; i < this.moves.length; i++) {
      const m = this.moves[i];
      tempChess.move(m);
      tempBoard.loadFen(tempChess.fen());
      const lastMoveObj = {
        from: (m.from.charCodeAt(1) - 49) * 8 + (m.from.charCodeAt(0) - 97),
        to: (m.to.charCodeAt(1) - 49) * 8 + (m.to.charCodeAt(0) - 97),
      };
      boosts.push(this.computePlyBoost(this.moves.slice(0, i + 1), lastMoveObj, tempBoard));
    }

    return boosts;
  }

  private updateAccretionHud(ledger: AccretionLedger): void {
    const list = this.container.querySelector('#accretion-ledger');
    if (!list) return;

    const entries = Object.entries(ledger.excessBySquare)
      .map(([sqStr, excess]) => ({ sq: parseInt(sqStr, 10), excess }))
      .filter((e) => this.board.squares[e.sq])
      .sort((a, b) => b.excess - a.excess)
      .slice(0, 5);

    if (!entries.length) {
      list.innerHTML = '<li class="accretion-empty">No captures yet — mass is conserved.</li>';
      return;
    }

    const GLYPHS: Record<string, { w: string; b: string }> = {
      p: { w: '♙', b: '♟' },
      n: { w: '♘', b: '♞' },
      b: { w: '♗', b: '♝' },
      r: { w: '♖', b: '♜' },
      q: { w: '♕', b: '♛' },
      k: { w: '♔', b: '♚' },
    };

    const consumedBySq: Record<number, { type: string; color: 'w' | 'b' }[]> = {};
    ledger.history.forEach((h) => {
      (consumedBySq[h.captorSquare] ??= []).push({ type: h.victimType, color: h.victimColor });
    });

    list.innerHTML = entries.map((e) => {
      const piece = this.board.squares[e.sq]!;
      const glyph = GLYPHS[piece.type]?.[piece.color] ?? piece.type;
      const total = piece.mass + e.excess;
      const consumed = (consumedBySq[e.sq] ?? [])
        .map((t) => GLYPHS[t.type]?.[t.color] ?? t.type)
        .join(', ');
      const fileChar = String.fromCharCode(97 + (e.sq % 8));
      const rank = Math.floor(e.sq / 8) + 1;
      return `<li class="accretion-item">
        <span>${glyph} ${fileChar}${rank} <strong>[${total.toFixed(1)}m]</strong></span>
        <span class="accretion-consumed">+${e.excess.toFixed(1)}m${consumed ? ` · ${consumed}` : ''}</span>
      </li>`;
    }).join('');
  }

  private updateTelemetry(breakdown: ScoreBreakdown): void {
    const move = this.moves[this.currentPlyIndex];
    const moveSan = move ? `${Math.floor(this.currentPlyIndex / 2) + 1}. ${move.color === 'b' ? '...' : ''}${move.san}` : 'Initial';
    const mover = this.board.turn === 'w' ? 'White' : 'Black';

    const moveEl = this.container.querySelector('#readout-move');
    if (moveEl) moveEl.textContent = moveSan;

    const moverEl = this.container.querySelector('#readout-mover');
    if (moverEl) moverEl.textContent = `${mover} to move`;

    // Vertical Barometer Update
    const barometerFill = this.container.querySelector('#barometer-fill') as HTMLElement;
    const barometerText = this.container.querySelector('#barometer-val-text');
    if (barometerFill) {
      const scoreW = breakdown.totalScoreWhite;
      const pct = Math.max(5, Math.min(95, 50 + (scoreW / 8.0) * 50));
      barometerFill.style.height = `${pct}%`;
      barometerFill.style.background = scoreW >= 0 ? 'var(--color-plate)' : 'var(--color-accent)';
    }
    if (barometerText) {
      barometerText.textContent = (breakdown.totalScoreWhite >= 0 ? '+' : '') + breakdown.totalScoreWhite.toFixed(1);
    }

    // Roche Status
    const statusEl = this.container.querySelector('#readout-status');
    const isUnderTidalStress = (breakdown.blackKingTidal?.isDisrupted || breakdown.whiteKingTidal?.isDisrupted);
    if (statusEl) {
      if (isUnderTidalStress) {
        statusEl.className = 'telemetry-status status-danger';
        statusEl.textContent = '⚠ ROCHE DISRUPTION IMMINENT';
      } else {
        statusEl.className = 'telemetry-status status-safe';
        statusEl.textContent = '● TIDAL EQUILIBRIUM';
      }
    }

    // King Roche Disruption Dial
    const activeKingTidal = this.board.turn === 'w' ? breakdown.blackKingTidal : breakdown.whiteKingTidal;
    const etaVal = activeKingTidal ? activeKingTidal.eta.toFixed(3) : '0.000';
    const rocheVal = this.config.roche.toFixed(2);
    const etaRatio = activeKingTidal ? Math.min(100, Math.round((activeKingTidal.eta / this.config.roche) * 100)) : 0;

    const etaText = this.container.querySelector('#gauge-eta-text');
    if (etaText) etaText.textContent = `η = ${etaVal} / ρ = ${rocheVal} (${etaRatio}%)`;

    const etaFill = this.container.querySelector('#gauge-eta-fill') as HTMLElement;
    if (etaFill) {
      etaFill.style.width = `${Math.min(100, etaRatio)}%`;
      etaFill.style.background = activeKingTidal?.isDisrupted ? 'var(--color-negative)' : 'var(--color-accent)';
    }

    // Stats band live Roche gauge
    const statsPct = this.container.querySelector('#stats-roche-pct');
    if (statsPct) statsPct.textContent = `${etaRatio}%`;

    const statsEta = this.container.querySelector('#stats-eta-text');
    if (statsEta) statsEta.textContent = `η = ${etaVal} / ρ = ${rocheVal}`;

    const statsFill = this.container.querySelector('#stats-roche-fill') as HTMLElement;
    if (statsFill) {
      statsFill.style.width = `${Math.min(100, etaRatio)}%`;
      statsFill.classList.toggle('danger', !!activeKingTidal?.isDisrupted);
    }

    // Score Waterfall Bars
    const updateBar = (id: string, val: number, max: number = 3.0) => {
      const row = this.container.querySelector(`#row-${id}`);
      if (!row) return;
      const valEl = row.querySelector('.val');
      const fillEl = row.querySelector('.waterfall-bar-fill') as HTMLElement;
      if (valEl) valEl.textContent = (val >= 0 ? '+' : '') + val.toFixed(2);
      if (fillEl) {
        const pct = Math.min(100, Math.round((Math.abs(val) / max) * 100));
        fillEl.style.width = `${pct}%`;
        fillEl.className = `waterfall-bar-fill ${val >= 0 ? 'fill-pos' : 'fill-neg'}`;
      }
    };

    updateBar('enemy-tide', breakdown.enemyKingTide, 2.5);
    updateBar('own-tide', breakdown.ownKingTide, 2.5);
    updateBar('enemy-force', breakdown.forceEnemyKing, 50.0);
    updateBar('own-force', breakdown.forceOwnKing, 50.0);
    updateBar('binding', breakdown.bindingEnergy, 1.5);
    updateBar('material', breakdown.materialBalance, 6.0);

    const totalEl = this.container.querySelector('#total-score-val');
    if (totalEl) {
      totalEl.textContent = (breakdown.totalScoreMover >= 0 ? '+' : '') + breakdown.totalScoreMover.toFixed(2) + ' native';
    }
  }

  // ---------- Play vs Kepler-64 ----------

  private updateModeDeckUI(): void {
    this.container.querySelectorAll('.mode-chip').forEach((chip) => {
      chip.classList.toggle('active', chip.getAttribute('data-mode') === this.mode);
    });

    const playDeck = this.container.querySelector('#play-deck') as HTMLElement | null;
    if (playDeck) playDeck.style.display = this.mode === 'play' ? 'flex' : 'none';

    this.container.querySelectorAll('.side-chip').forEach((chip) => {
      chip.classList.toggle('active', chip.getAttribute('data-side') === this.playerColor);
    });

    const select = this.container.querySelector('#persona-select') as HTMLSelectElement | null;
    if (select) select.value = this.personaId;

    this.updatePersonaBlurb();

    const transportIds = ['#btn-first', '#btn-prev', '#btn-play', '#btn-next', '#btn-last', '#ply-slider'];
    transportIds.forEach((id) => {
      const el = this.container.querySelector(id) as HTMLButtonElement | HTMLInputElement | null;
      if (!el) return;
      if (this.mode === 'play') el.setAttribute('disabled', 'true');
      else el.removeAttribute('disabled');
    });
  }

  private updatePersonaBlurb(): void {
    const el = this.container.querySelector('#persona-blurb');
    if (el) el.textContent = getPersona(this.personaId).blurb;
  }

  private updateBotStatus(text: string): void {
    const el = this.container.querySelector('#bot-status-text');
    if (el) el.textContent = text;
    const root = this.container.querySelector('#bot-status');
    if (root) root.classList.toggle('thinking', this.isBotThinking);
  }

  private syncConfigToUi(): void {
    const map: Array<[string, keyof ConstantsConfig, string]> = [
      ['g', 'G', ''],
      ['eps', 'eps', ''],
      ['c', 'c', ' sq/ply'],
      ['roche', 'roche', ''],
    ];
    map.forEach(([id, key, unit]) => {
      const el = this.container.querySelector(`#slider-${id}`) as HTMLInputElement | null;
      const valEl = this.container.querySelector(`#slider-${id}-val`);
      if (el) el.value = this.config[key].toString();
      if (valEl) valEl.textContent = `${this.config[key].toFixed(2)}${unit}`;
    });
  }

  private sqToAlg(sq: number): string {
    return String.fromCharCode(97 + (sq % 8)) + (Math.floor(sq / 8) + 1);
  }

  private algToSq(alg: string): number {
    const file = alg.charCodeAt(0) - 97;
    const rank = alg.charCodeAt(1) - 49;
    return rank * 8 + file;
  }

  private findKingSquare(color: 'w' | 'b'): number | null {
    for (let sq = 0; sq < 64; sq++) {
      const piece = this.board.squares[sq];
      if (piece && piece.type === 'k' && piece.color === color) {
        return sq;
      }
    }
    return null;
  }
  private startPlayMode(side: 'w' | 'b', personaId: string): void {
    if (this.isPlaying) this.toggleAutoplay();
    if (this.sparklineDebounce) {
      window.clearTimeout(this.sparklineDebounce);
      this.sparklineDebounce = undefined;
    }
    this.cleanupWorker();
    this.cleanupMultiverseWorker();

    if (this.mode === 'replay') {
      this.replayConfig = { ...this.config };
    }
    this.mode = 'play';
    this.playerColor = side;
    this.personaId = personaId;
    this.isBotThinking = false;
    this.liveChess = new Chess();
    this.startFen = null;
    this.moves = [];
    this.currentPlyIndex = 0;

    if (this.canvasRenderer) this.canvasRenderer.setOrientation(side);

    // Enter the persona's physical universe.
    this.config = { ...getPersona(personaId).config };
    this.syncConfigToUi();
    if (this.canvasRenderer) this.canvasRenderer.setConfig(this.config);

    this.syncBoardToPly(0);
    this.updateSparklineData();

    const slider = this.container.querySelector('#ply-slider') as HTMLInputElement | null;
    if (slider) {
      slider.max = '0';
      slider.value = '0';
    }
    const plyIndicator = this.container.querySelector('#ply-indicator');
    if (plyIndicator) plyIndicator.textContent = 'Ready';

    this.updateModeDeckUI();

    if (this.liveChess.turn() === this.playerColor) {
      this.updateBotStatus('Your move');
    } else {
      this.updateBotStatus('Kepler-64 moves first…');
      this.scheduleBotMove();
    }
  }

  private switchToReplay(): void {
    this.cleanupWorker();
    this.isBotThinking = false;
    this.initGame(this.currentGame);
  }

  private handleImportPgn(pgn: string): void {
    const chess = new Chess();
    chess.loadPgn(pgn);
    const headers = chess.header();
    const white = headers.White ?? 'White';
    const black = headers.Black ?? 'Black';
    const fenHeader = (headers.FEN ?? '').trim();
    const headerFen = fenHeader.split(/\s+/).length === 6
      ? fenHeader
      : 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

    const synthetic: PresetGame = {
      id: 'imported-pgn',
      title: headers.White || headers.Black
        ? `${white} vs ${black}`
        : 'Imported Game',
      subtitle: 'Custom PGN analysis',
      white,
      black,
      date: headers.Date ?? '',
      event: headers.Event ?? '',
      initialFen: headerFen,
      highlightPly: Math.max(0, chess.history().length - 1),
      pgn,
    };

    this.container.querySelectorAll('.preset-pill').forEach((p) => p.classList.remove('active'));
    this.initGame(synthetic);
  }

  private handleImportFen(fen: string): void {
    this.cleanupWorker();
    this.cleanupMultiverseWorker();
    if (this.sparklineDebounce) {
      window.clearTimeout(this.sparklineDebounce);
      this.sparklineDebounce = undefined;
    }
    this.isBotThinking = false;
    this.mode = 'replay';
    this.config = { ...this.replayConfig };
    this.syncConfigToUi();
    if (this.canvasRenderer) this.canvasRenderer.setConfig(this.config);
    this.startFen = fen;
    this.moves = [];
    this.currentPlyIndex = 0;

    if (this.canvasRenderer) this.canvasRenderer.setOrientation('w');

    this.currentGame = {
      id: 'imported-fen',
      title: 'Imported Position',
      subtitle: 'Custom FEN analysis',
      white: 'White',
      black: 'Black',
      date: '',
      event: '',
      initialFen: fen,
      highlightPly: 0,
      pgn: '',
    };

    this.board.loadFen(fen);
    if (this.canvasRenderer) this.canvasRenderer.setBoard(this.board, null);
    this.updateCandidateMoves();
    this.updateSparklineData();

    const slider = this.container.querySelector('#ply-slider') as HTMLInputElement | null;
    if (slider) {
      slider.max = '0';
      slider.value = '0';
    }
    const plyIndicator = this.container.querySelector('#ply-indicator');
    if (plyIndicator) plyIndicator.textContent = 'Position';

    this.container.querySelectorAll('.preset-pill').forEach((p) => p.classList.remove('active'));
    this.updateModeDeckUI();
  }

  private handleUserMove(fromSq: number, toSq: number): void {
    if (this.isBotThinking) return;

    if (this.mode === 'play') {
      if (this.liveChess.turn() !== this.playerColor) return;

      const from = this.sqToAlg(fromSq);
      const to = this.sqToAlg(toSq);
      const legal = this.liveChess.moves({ verbose: true });
      const matching = legal.filter((m) => m.from === from && m.to === to);
      const move = matching.find((m) => m.promotion === 'q')
        ?? matching.find((m) => !m.promotion)
        ?? matching[0];
      if (!move) return;

      let applied = false;
      try {
        applied = !!this.liveChess.move({ from, to, promotion: move.promotion });
      } catch {
        applied = false;
      }
      if (!applied) return;

      this.refreshLiveGame();
      this.maybeScheduleBot();
    } else {
      // In Replay / Analysis mode: Allow user to freely play moves from the current position!
      const currentFen = this.board.toFen();
      let tempChess: Chess;
      try {
        tempChess = new Chess(currentFen);
      } catch {
        tempChess = new Chess();
      }

      const from = this.sqToAlg(fromSq);
      const to = this.sqToAlg(toSq);
      const legal = tempChess.moves({ verbose: true });
      const matching = legal.filter((m) => m.from === from && m.to === to);
      const move = matching.find((m) => m.promotion === 'q')
        ?? matching.find((m) => !m.promotion)
        ?? matching[0];
      if (!move) return;

      let applied = false;
      try {
        applied = !!tempChess.move({ from, to, promotion: move.promotion });
      } catch {
        applied = false;
      }
      if (!applied) return;

      // Automatically branch into interactive play from this position
      this.mode = 'play';
      this.liveChess = tempChess;
      this.playerColor = tempChess.turn() === 'w' ? 'b' : 'w';
      this.moves = this.liveChess.history({ verbose: true });
      this.currentPlyIndex = Math.max(0, this.moves.length - 1);
      this.updateModeDeckUI();
      this.refreshLiveGame();
      this.maybeScheduleBot();
    }
  }

  private refreshLiveGame(): void {
    this.moves = this.liveChess.history({ verbose: true });
    this.currentPlyIndex = Math.max(0, this.moves.length - 1);
    this.syncBoardToPly(this.currentPlyIndex);
    this.updateSparklineData();

    const last = this.moves[this.moves.length - 1];
    if (last?.captured) {
      const fromSq = (last.from.charCodeAt(1) - 49) * 8 + (last.from.charCodeAt(0) - 97);
      const toSq = (last.to.charCodeAt(1) - 49) * 8 + (last.to.charCodeAt(0) - 97);
      this.canvasRenderer.playCaptureStream(fromSq, toSq);
    }

    const slider = this.container.querySelector('#ply-slider') as HTMLInputElement | null;
    if (slider) {
      slider.max = Math.max(0, this.moves.length - 1).toString();
      slider.value = this.currentPlyIndex.toString();
    }
    const plyIndicator = this.container.querySelector('#ply-indicator');
    if (plyIndicator) {
      plyIndicator.textContent = this.moves.length
        ? `Ply ${this.currentPlyIndex + 1} / ${this.moves.length}`
        : 'Ready';
    }
  }

  private maybeScheduleBot(): void {
    if (this.mode !== 'play') return;
    if (this.liveChess.isGameOver()) {
      this.updateBotStatus(this.gameOverText());
      return;
    }
    if (this.liveChess.turn() === this.playerColor) {
      this.updateBotStatus('Your move');
      return;
    }
    this.scheduleBotMove();
  }

  private gameOverText(): string {
    if (this.liveChess.isCheckmate()) {
      const winner = this.liveChess.turn() === 'w' ? 'Black' : 'White';
      return `${winner} wins by gravitational collapse`;
    }
    if (this.liveChess.isStalemate()) return 'Stalemate — tidal equilibrium';
    if (this.liveChess.isDraw()) return 'Draw — tidal equilibrium';
    return 'Game over';
  }

  private scheduleBotMove(): void {
    this.isBotThinking = true;
    this.updateBotStatus('Kepler-64 is thinking…');

    this.searchGen++;
    const gen = this.searchGen;

    const worker = new Worker(new URL('../core/searchWorker.ts', import.meta.url), { type: 'module' });
    this.searchWorker = worker;

    worker.onmessage = (e: MessageEvent) => {
      if (gen !== this.searchGen) return;
      this.cleanupWorker();
      this.isBotThinking = false;

      const data = e.data as { ok?: boolean; result?: SearchResult; error?: string } | undefined;
      if (!data?.ok || !data.result) {
        this.updateBotStatus(data?.error ? `Search error: ${data.error}` : 'Search failed');
        return;
      }

      const mv = data.result;
      let applied = false;
      try {
        applied = !!this.liveChess.move({ from: mv.from, to: mv.to, promotion: mv.promotion });
      } catch {
        applied = false;
      }
      if (!applied) {
        this.updateBotStatus('Illegal bot move rejected');
        return;
      }

      this.refreshLiveGame();
      if (this.liveChess.isGameOver()) {
        this.updateBotStatus(`${this.gameOverText()} · ${mv.san}`);
      } else {
        this.updateBotStatus(`Kepler-64 played ${mv.san} · ${mv.note}`);
      }
    };

    worker.onerror = () => {
      if (gen !== this.searchGen) return;
      this.cleanupWorker();
      this.isBotThinking = false;
      this.updateBotStatus('Search worker crashed');
    };

    worker.postMessage({
      fen: this.liveChess.fen(),
      personaId: this.personaId,
      baseExcess: this.accretionExcess,
    });
  }

  private cleanupWorker(): void {
    if (this.searchWorker) {
      this.searchWorker.terminate();
      this.searchWorker = null;
    }
    this.searchGen += 1;
  }

  private cleanupMultiverseWorker(): void {
    if (this.multiverseWorker) {
      this.multiverseWorker.terminate();
      this.multiverseWorker = null;
    }
    this.multiverseGen += 1;
  }

  private toggleAutoplay(): void {
    if (this.isPlaying) {
      this.isPlaying = false;
      if (this.playIntervalId) clearInterval(this.playIntervalId);
      const playBtn = this.container.querySelector('#btn-play');
      if (playBtn) playBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        Play
      `;
    } else {
      if (this.mode === 'play') {
        this.switchToReplay();
      }
      if (this.currentPlyIndex >= this.moves.length - 1) {
        this.setPly(0);
      }
      this.isPlaying = true;
      const playBtn = this.container.querySelector('#btn-play');
      if (playBtn) playBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
        Pause
      `;
      this.playIntervalId = window.setInterval(() => {
        if (this.currentPlyIndex < this.moves.length - 1) {
          this.setPly(this.currentPlyIndex + 1);
        } else {
          this.toggleAutoplay();
        }
      }, 1000);
    }
  }

  private setPly(index: number): void {
    if (this.mode === 'play') return;
    this.currentPlyIndex = Math.max(0, Math.min(this.moves.length - 1, index));
    this.syncBoardToPly(this.currentPlyIndex);

    const slider = this.container.querySelector('#ply-slider') as HTMLInputElement;
    if (slider) slider.value = this.currentPlyIndex.toString();

    const plyIndicator = this.container.querySelector('#ply-indicator');
    if (plyIndicator) plyIndicator.textContent = `Ply ${this.currentPlyIndex + 1} / ${this.moves.length}`;
  }

  private render(): void {
    this.container.innerHTML = `
      <!-- FLOATING PILL NAV -->
      <header class="pill-nav-wrap">
        <div class="pill-nav shell">
          <a href="#top" class="brand">
            <span>KEPLER-64</span>
            <span class="brand-badge">ROCHE ENGINE</span>
          </a>

          <nav class="stacked-nav" aria-label="Primary">
            <a href="#observatory" class="stack-link">Observatory</a>
            <a href="#compendium" class="stack-link">Compendium</a>
            <a href="#mechanism" class="stack-link">How it Works</a>
            <a href="#sandbox" class="stack-link">Laboratory</a>
            <a href="#contributors" class="stack-link">Contributors</a>
          </nav>

          <div class="nav-actions">
            <button id="nav-export-btn" class="action-primary nav-export-btn" title="Export Social GIF">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
              <span>Export GIF</span>
            </button>
            <a href="https://github.com/r-baruah/kepler-64" target="_blank" rel="noreferrer" class="action-secondary star-btn" title="Star Kepler-64 on GitHub">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              <span>★ Star</span>
            </a>
          </div>
        </div>
      </header>

      <!-- HERO -->
      <section class="hero shell" id="top">
        <div class="hero-stage">
          <span class="hero-watermark" aria-hidden="true">KEPLER</span>

          <aside class="hero-accent-card">
            <div class="hero-accent-brand">KEPLER-64<sup>®</sup></div>
            <p>We invest mass, tidal tensors and differentiable engineering into every position worth evaluating.</p>
            <p>From idea to King collapse in under 90 plies.</p>
            <a href="#observatory" class="hero-accent-cta">Details →</a>
            <div class="hero-accent-est"><span>Est.</span><strong>2026</strong></div>
          </aside>

          <div class="hero-head">
            <h1 class="hero-headline">
              <span>Forging</span>
              <span>gravity</span>
              <span>that</span>
              <span>plays chess.</span>
            </h1>
            <div class="hero-cta-row">
              <a href="#observatory" class="hero-arrow-btn" aria-label="Open the Observatory">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="square" stroke-linejoin="miter" aria-hidden="true">
                  <path d="M4 12h16"></path>
                  <path d="M13 5l7 7-7 7"></path>
                </svg>
              </a>
              <span class="hero-cta-text">Open the Observatory</span>
            </div>
          </div>
        </div>

      </section>

      <!-- STATS BAND -->
      <section class="stats-band shell" id="numbers">
        <div class="section-head">
          <span class="section-number">01 — By the numbers</span>
          <h2>What the engine learns.</h2>
          <p class="section-lead">Every physical constant is a learnable leaf. The evaluation is a gradient, not a guess.</p>
        </div>

        <div class="stats-grid">
          <div class="goal-card">
            <div class="kpi-eyebrow">PRIMARY GOAL</div>
            <h3>Differentiable Evaluation</h3>
            <div class="goal-figure">
              <span class="big-stat">100<sup>%</sup></span>
              <span class="goal-caption">end-to-end JAX auto-diff</span>
            </div>
            <div class="progress-track"><div class="progress-fill" style="width:100%;"></div></div>
            <p class="kpi-note">13 of 13 physical constants are learnable leaves, optimized by gradient descent over self-play.</p>
          </div>

          <div class="live-gauge-card">
            <div class="kpi-eyebrow">ROCHE DISRUPTION INDEX</div>
            <div class="live-gauge-row">
              <span class="big-stat" id="stats-roche-pct">0%</span>
              <span class="goal-caption" id="stats-eta-text">η = 0.000 / ρ = ${this.config.roche.toFixed(2)}</span>
            </div>
            <div class="progress-track warm"><div id="stats-roche-fill" class="progress-fill warm" style="width:0%;"></div></div>
            <p class="kpi-note">The enemy King collapses when η &gt; ρ<sub>roche</sub>.</p>
          </div>
        </div>

        <div class="kpi-grid">
          <div class="kpi-card"><span class="kpi-label">LEARNABLE LEAVES</span><span class="kpi-value">13</span><span class="kpi-change">13 / 13 trained</span></div>
          <div class="kpi-card"><span class="kpi-label">LATTICE</span><span class="kpi-value">64×64</span><span class="kpi-change">2D spacetime</span></div>
          <div class="kpi-card"><span class="kpi-label">MULTIVERSE</span><span class="kpi-value">K = 8</span><span class="kpi-change">posterior worlds</span></div>
          <div class="kpi-card"><span class="kpi-label">ROCHE LIMIT</span><span class="kpi-value">ρ 0.80</span><span class="kpi-change">rupture threshold</span></div>
        </div>
      </section>

      <!-- OBSERVATORY CONSOLE -->
      <section class="observatory-console-section shell" id="observatory">
        <div class="section-head compact">
          <span class="section-number">02 — Observatory</span>
          <h2>Watch gravity play.</h2>
        </div>

        <!-- PRESET SELECTOR -->
        <div class="preset-bar">
          <span class="preset-label">OBSERVATIONS:</span>
          ${PRESET_GAMES.map((g, idx) => `
            <button class="preset-pill ${idx === 0 ? 'active' : ''}" data-preset-id="${g.id}">
              ${g.title}
            </button>
          `).join('')}
          <button id="btn-import-game" class="preset-pill import-pill">Import Game</button>
        </div>

        <div class="mode-deck">
          <div class="mode-switch">
            <span class="deck-label">MODE</span>
            <button class="mode-chip active" data-mode="replay">▶ Replay</button>
            <button class="mode-chip" data-mode="play">Play vs Kepler</button>
          </div>

          <div id="play-deck" class="play-deck" style="display:none;">
            <span class="deck-label">SIDE</span>
            <button class="side-chip active" data-side="w">White</button>
            <button class="side-chip" data-side="b">Black</button>

            <span class="deck-label">PERSONA</span>
            <select id="persona-select" class="persona-select">
              ${BOT_PERSONAS.map((p) => `<option value="${p.id}">${p.label}</option>`).join('')}
            </select>

            <button id="btn-new-game" class="action-secondary new-game-btn">↺ New Game</button>

            <span id="bot-status" class="bot-status">
              <span class="bot-status-dot"></span>
              <span id="bot-status-text"></span>
            </span>

            <span id="persona-blurb" class="persona-blurb"></span>
          </div>
        </div>

        <div class="observatory-cockpit-grid">
          <!-- LEFT: FUSED BOARD INSTRUMENT + PLAYBACK -->
          <div class="board-column">
            <div class="board-with-barometer">
              <div class="vertical-barometer" title="Net Spacetime Curvature (White vs Black)">
                <div class="barometer-track">
                  <div id="barometer-fill" class="barometer-fill"></div>
                </div>
                <span id="barometer-val-text" class="barometer-label">+0.0</span>
              </div>

              <div class="board-container">
                <canvas id="board-canvas" class="board-canvas"></canvas>
              </div>
            </div>

            <div class="transport-bar">
              <div class="transport-buttons">
                <button id="btn-first" class="action-secondary transport-btn" title="First Move">⏮</button>
                <button id="btn-prev" class="action-secondary transport-btn" title="Previous Move">◀</button>
                <button id="btn-play" class="action-primary transport-play-btn">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                  Play
                </button>
                <button id="btn-next" class="action-secondary transport-btn" title="Next Move">▶</button>
                <button id="btn-last" class="action-secondary transport-btn" title="Last Move">⏭</button>
              </div>

              <div class="scrubber-row">
                <input id="ply-slider" type="range" min="0" max="${Math.max(0, this.moves.length - 1)}" value="${this.currentPlyIndex}" class="scrubber-slider" />
                <strong id="ply-indicator" class="ply-indicator">Ply ${this.currentPlyIndex + 1} / ${this.moves.length}</strong>
                <button id="btn-export-quick" class="action-secondary export-quick-btn" title="Export Social GIF">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                  GIF
                </button>
              </div>
            </div>

            <div class="board-footer-strip">
              <div class="hover-hud">
                <span id="hover-hud-text">Tap or hover any square to inspect local force &amp; mass</span>
              </div>
              <div class="layer-toggles">
                <button class="toggle-chip active" data-layer="showContours">Contours</button>
                <button class="toggle-chip active" data-layer="showHeatmap">Potential</button>
                <button class="toggle-chip" data-layer="showVectors">Vectors</button>
                <button class="toggle-chip active" data-layer="showTidalStress">Tidal Tensors</button>
                <button class="toggle-chip active" data-layer="showAccretion">Accretion</button>
                <button class="toggle-chip" data-layer="showWavefronts">Light Cone</button>
                <button class="toggle-chip" data-layer="showLorentz">Kinetic</button>
                <button class="toggle-chip" data-layer="showLagrange">Lagrange</button>
                <button class="toggle-chip" data-layer="multiverse">Multiverse</button>
              </div>
            </div>
          </div>

          <!-- RIGHT: TRAJECTORY + TELEMETRY -->
          <div class="telemetry-column">
            <div id="sparkline-container" class="cockpit-sparkline-wrap"></div>
            <div class="telemetry-card">
              <div class="telemetry-header">
                <div>
                  <div id="readout-move" class="telemetry-move">—</div>
                  <div id="readout-mover" class="telemetry-mover">White to move</div>
                </div>
                <div id="readout-status" class="telemetry-status status-safe">● TIDAL EQUILIBRIUM</div>
              </div>

              <div class="roche-gauge-box">
                <div class="gauge-header">
                  <span>King Roche Disruption Index (η)</span>
                  <span id="gauge-eta-text">η = 0.000 / ρ = ${this.config.roche.toFixed(2)}</span>
                </div>
                <div class="roche-track">
                  <div id="gauge-eta-fill" class="roche-fill" style="width:15%;"></div>
                </div>
              </div>

              <div class="telemetry-dual-grid">
                <div>
                  <div class="pane-subtitle">SCORE DECOMPOSITION</div>
                  <ul class="waterfall-ledger">
                    <li id="row-enemy-tide" class="waterfall-row">
                      <span>Enemy King Tide</span>
                      <div class="waterfall-bar-track"><div class="waterfall-bar-fill fill-pos" style="width:0%;"></div></div>
                      <span class="val">+0.00</span>
                    </li>
                    <li id="row-own-tide" class="waterfall-row">
                      <span>Own King Tide</span>
                      <div class="waterfall-bar-track"><div class="waterfall-bar-fill fill-neg" style="width:0%;"></div></div>
                      <span class="val">-0.00</span>
                    </li>
                    <li id="row-enemy-force" class="waterfall-row">
                      <span>Disruption Force</span>
                      <div class="waterfall-bar-track"><div class="waterfall-bar-fill fill-pos" style="width:0%;"></div></div>
                      <span class="val">+0.00</span>
                    </li>
                    <li id="row-binding" class="waterfall-row">
                      <span>Binding Energy (ΔE)</span>
                      <div class="waterfall-bar-track"><div class="waterfall-bar-fill fill-pos" style="width:0%;"></div></div>
                      <span class="val">+0.00</span>
                    </li>
                    <li id="row-material" class="waterfall-row">
                      <span>Material Edge</span>
                      <div class="waterfall-bar-track"><div class="waterfall-bar-fill fill-pos" style="width:0%;"></div></div>
                      <span class="val">+0.00</span>
                    </li>
                  </ul>
                </div>

                <div class="candidates-pane">
                  <div class="candidates-pane-header">
                    <span>TOP CANDIDATES</span>
                    <span>Score</span>
                  </div>
                  <ul id="candidate-moves-list" class="candidate-list">
                    <!-- Populated dynamically -->
                  </ul>
                </div>
              </div>

              <div class="net-score-bar">
                <span class="net-score-label">NET POSITION SCORE:</span>
                <span id="total-score-val" class="net-score-val">+0.00 native</span>
              </div>

              <div class="accretion-pane">
                <div class="pane-subtitle">ACCRETION LEDGER</div>
                <ul id="accretion-ledger" class="accretion-list">
                  <li class="accretion-empty">No captures yet — mass is conserved.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- COMPENDIUM -->
      <section class="compendium-band" id="compendium-section">
        <div id="field-guide-mount"></div>
      </section>

      <!-- MECHANISM PLATE -->
      <section class="mechanism-plate" id="mechanism">
        <div class="shell">
          <span class="section-number">03 — How it works</span>
          <div class="badge-tag">ANALYTICAL ENGINE</div>
          <h2>The physics is the evaluator.</h2>
          <p class="section-lead" style="opacity:0.9; max-width:65ch; margin-top:var(--space-sm);">
            Kepler-64 evaluates every legal move through an explicit gravitational and tidal tensor calculation.
            Every constant of this universe is a learnable leaf optimized through JAX gradient descent.
          </p>

          <div class="equation-box">
            <div class="equation-label">1. PLUMMER POTENTIAL FIELD</div>
            <div>${latex('\\Phi(p) = -G \\sum_{j=1}^{64} \\frac{|m_j| \\cdot \\sigma(c - \\lVert p - r_j \\rVert)}{\\sqrt{\\lVert p - r_j \\rVert^2 + \\varepsilon^2}}', true)}</div>

            <div class="equation-label">2. TIDAL TENSOR (HESSIAN) &amp; LINE OF FAILURE</div>
            <div>${latex('\\mathbf{A} = \\nabla\\nabla\\Phi\\big|_{\\text{King}} = \\begin{pmatrix} \\Phi_{xx} & \\Phi_{xy} \\\\ \\Phi_{yx} & \\Phi_{yy} \\end{pmatrix} \\quad \\implies \\quad \\lambda_1, \\lambda_2, \\; \\vec{v}_1', true)}</div>

            <div class="equation-label">3. ROCHE DISRUPTION WIN CRITERION</div>
            <div>${latex('\\eta = \\frac{R_g^3 \\cdot \\lambda_1}{m_{\\text{ref}}^2} \\quad \\implies \\quad \\text{King collapses when } \\eta > \\rho_{\\text{roche}}', true)}</div>
          </div>
        </div>
      </section>

      <!-- SANDBOX -->
      <section class="shell sandbox-section" id="sandbox">
        <div class="section-head compact">
          <span class="section-number">04 — Laboratory</span>
          <h2>Universal Physics Laboratory</h2>
          <p class="section-lead" style="max-width:65ch;">
            Adjust the fundamental physical laws of the chess universe in real time and watch the gravitational topography and tidal tension warp instantly.
          </p>
        </div>

        <div class="sandbox-slider-grid">
          <div class="slider-group">
            <div class="slider-header">
              <span>Gravitational Constant (G)</span>
              <span id="slider-g-val">${this.config.G.toFixed(2)}</span>
            </div>
            <input id="slider-g" type="range" min="0.1" max="4.0" step="0.05" value="${this.config.G}" />
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Plummer Softening (ε)</span>
              <span id="slider-eps-val">${this.config.eps.toFixed(2)}</span>
            </div>
            <input id="slider-eps" type="range" min="0.1" max="2.5" step="0.05" value="${this.config.eps}" />
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Speed of Light Gate (c)</span>
              <span id="slider-c-val">${this.config.c.toFixed(2)} sq/ply</span>
            </div>
            <input id="slider-c" type="range" min="1.0" max="10.0" step="0.1" value="${this.config.c}" />
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Roche Critical Limit (ρ)</span>
              <span id="slider-roche-val">${this.config.roche.toFixed(2)}</span>
            </div>
            <input id="slider-roche" type="range" min="0.1" max="2.0" step="0.05" value="${this.config.roche}" />
          </div>
        </div>
      </section>

      <!-- CONTRIBUTORS -->
      <section class="shell">
        <div id="contributors-mount"></div>
      </section>

      <!-- FOOTER -->
      <footer class="site-footer">
        <div class="footer-wordmark" aria-hidden="true">KEPLER-64</div>
        <div class="shell footer-inner">
          <div class="footer-meta">
            <a href="#top" class="brand">
              <span>KEPLER-64</span>
              <span class="brand-badge">ROCHE ENGINE</span>
            </a>
            <div class="footer-note">
              Kepler-64 © 2026 · Created by <strong>Ripuranjan Baruah</strong> · Open-Source Research
            </div>
          </div>
          <div class="footer-links">
            <a href="https://github.com/r-baruah/kepler-64" target="_blank" rel="noreferrer" class="action-secondary">GitHub Repository</a>
            <a href="https://github.com/r-baruah/kepler-64/blob/main/REFERENCES.md" target="_blank" rel="noreferrer" class="action-secondary">References</a>
            <a href="https://github.com/r-baruah/kepler-64/issues" target="_blank" rel="noreferrer" class="action-secondary">Report Issue</a>
          </div>
        </div>
      </footer>
    `;

    this.attachEvents();
    this.updateModeDeckUI();
  }

  private attachEvents(): void {
    // Preset game switcher
    this.container.querySelectorAll('.preset-pill').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const id = (e.currentTarget as HTMLElement).getAttribute('data-preset-id');
        const match = PRESET_GAMES.find((g) => g.id === id);
        if (match) {
          this.container.querySelectorAll('.preset-pill').forEach((p) => p.classList.remove('active'));
          (e.currentTarget as HTMLElement).classList.add('active');
          this.initGame(match);
        }
      });
    });

    // Layer toggles
    this.container.querySelectorAll('.toggle-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        const el = e.currentTarget as HTMLElement;
        const layer = el.getAttribute('data-layer') as any;
        const isActive = el.classList.toggle('active');
        if (layer === 'multiverse') {
          this.multiverseEnabled = isActive;
          this.updateSparklineData();
          return;
        }
        if (this.canvasRenderer && layer) {
          this.canvasRenderer.setLayers({ [layer]: isActive });
        }
      });
    });

    // Mode deck: replay vs play-vs-Kepler
    this.container.querySelectorAll('.mode-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        const mode = (e.currentTarget as HTMLElement).getAttribute('data-mode');
        if (mode === 'play') {
          this.startPlayMode(this.playerColor, this.personaId);
        } else {
          this.switchToReplay();
        }
      });
    });

    this.container.querySelectorAll('.side-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        const side = (e.currentTarget as HTMLElement).getAttribute('data-side') as 'w' | 'b';
        if (this.mode === 'play') {
          this.startPlayMode(side, this.personaId);
        } else {
          this.playerColor = side;
          this.updateModeDeckUI();
        }
      });
    });

    this.container.querySelector('#persona-select')?.addEventListener('change', (e) => {
      this.personaId = (e.target as HTMLSelectElement).value;
      this.updatePersonaBlurb();
      if (this.mode === 'play') {
        this.startPlayMode(this.playerColor, this.personaId);
      }
    });

    this.container.querySelector('#btn-new-game')?.addEventListener('click', () => {
      this.startPlayMode(this.playerColor, this.personaId);
    });

    // Transport buttons
    this.container.querySelector('#btn-first')?.addEventListener('click', () => {
      this.setPly(0);
    });

    this.container.querySelector('#btn-last')?.addEventListener('click', () => {
      this.setPly(this.moves.length - 1);
    });

    this.container.querySelector('#btn-prev')?.addEventListener('click', () => {
      if (this.currentPlyIndex > 0) this.setPly(this.currentPlyIndex - 1);
    });

    this.container.querySelector('#btn-next')?.addEventListener('click', () => {
      if (this.currentPlyIndex < this.moves.length - 1) this.setPly(this.currentPlyIndex + 1);
    });

    this.container.querySelector('#btn-play')?.addEventListener('click', () => {
      this.toggleAutoplay();
    });

    const slider = this.container.querySelector('#ply-slider') as HTMLInputElement;
    slider?.addEventListener('input', (e) => {
      const idx = parseInt((e.target as HTMLInputElement).value, 10);
      this.setPly(idx);
    });

    // Modal Triggers
    this.container.querySelector('#nav-export-btn')?.addEventListener('click', () => {
      this.exportModal.open(this.currentGame, this.config);
    });
    this.container.querySelector('#btn-export-quick')?.addEventListener('click', () => {
      this.exportModal.open(this.currentGame, this.config);
    });

    this.container.querySelector('#btn-import-game')?.addEventListener('click', () => {
      this.importModal.open();
    });

    // Sliders
    const bindSlider = (id: string, key: keyof ConstantsConfig, unit: string = '') => {
      const el = this.container.querySelector(`#slider-${id}`) as HTMLInputElement;
      const valEl = this.container.querySelector(`#slider-${id}-val`);
      el?.addEventListener('input', (e) => {
        const val = parseFloat((e.target as HTMLInputElement).value);
        this.config[key] = val;
        if (valEl) valEl.textContent = `${val.toFixed(2)}${unit}`;
        if (this.canvasRenderer) {
          this.canvasRenderer.setConfig(this.config);
        }
        if (this.sparkline) {
          this.scheduleSparklineRefresh();
        }
      });
    };

    bindSlider('g', 'G');
    bindSlider('eps', 'eps');
    bindSlider('c', 'c', ' sq/ply');
    bindSlider('roche', 'roche');

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
      if (this.mode === 'play') return;
      if (e.key === 'ArrowLeft') {
        if (this.currentPlyIndex > 0) this.setPly(this.currentPlyIndex - 1);
      } else if (e.key === 'ArrowRight') {
        if (this.currentPlyIndex < this.moves.length - 1) this.setPly(this.currentPlyIndex + 1);
      } else if (e.key === ' ') {
        e.preventDefault();
        this.toggleAutoplay();
      }
    });
  }
}
