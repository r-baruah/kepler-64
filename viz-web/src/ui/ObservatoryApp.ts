/**
 * Kepler-64 Observatory Master Application Component
 * State-of-the-Art Astrophysical Chess Observatory Console
 */

import { Chess } from 'chess.js';
import { KeplerBoard } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { DEFAULT_CONSTANTS } from '../core/constants';
import { PRESET_GAMES } from '../core/presets';
import type { PresetGame } from '../core/presets';
import { UnifiedCanvas } from '../render/UnifiedCanvas';
import type { ScoreBreakdown } from '../core/evaluate';
import { evaluatePosition } from '../core/evaluate';
import { ExportModal } from './ExportModal';
import { EvalSparkline } from './EvalSparkline';
import { FieldGuideComponent } from './FieldGuide';
import { ContributorsSection } from './ContributorsSection';
import { latex } from './katexUtil';

export class ObservatoryApp {
  private container: HTMLElement;
  private config: ConstantsConfig = { ...DEFAULT_CONSTANTS };
  private currentGame: PresetGame = PRESET_GAMES[0];
  private chess: Chess = new Chess();
  private board: KeplerBoard = new KeplerBoard();
  private moves: any[] = [];
  private currentPlyIndex = 0;
  private isPlaying = false;
  private playIntervalId?: number;

  private canvasRenderer!: UnifiedCanvas;
  private exportModal!: ExportModal;
  private sparkline!: EvalSparkline;
  private fieldGuide!: FieldGuideComponent;
  private contributors!: ContributorsSection;

  constructor(container: HTMLElement) {
    this.container = container;
    this.initGame(this.currentGame);
    this.render();
    this.initCanvas();
    this.initSparkline();
    this.initFieldGuide();
    this.initContributors();
    this.exportModal = new ExportModal(this.currentGame, this.config);
  }

  private initGame(game: PresetGame): void {
    this.currentGame = game;
    this.chess = new Chess();
    this.chess.loadPgn(game.pgn);
    this.moves = this.chess.history({ verbose: true });
    this.currentPlyIndex = Math.min(game.highlightPly, this.moves.length - 1);
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
  }

  private syncBoardToPly(plyIndex: number): void {
    const tempChess = new Chess();
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
    if (this.canvasRenderer) {
      this.canvasRenderer.setBoard(this.board, lastMoveObj);
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
    const tempChess = new Chess();
    const tempBoard = new KeplerBoard();
    const points: { ply: number; score: number; moveSan: string }[] = [];

    for (let i = 0; i < this.moves.length; i++) {
      const m = this.moves[i];
      tempChess.move(m);
      tempBoard.loadFen(tempChess.fen());
      const evalRes = evaluatePosition(tempBoard, this.config);
      points.push({
        ply: i,
        score: evalRes.totalScoreWhite,
        moveSan: m.san,
      });
    }
    this.sparkline.setData(points, this.currentPlyIndex);
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
      etaFill.style.background = activeKingTidal?.isDisrupted ? 'var(--color-negative)' : 'var(--color-plate)';
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
    this.currentPlyIndex = Math.max(0, Math.min(this.moves.length - 1, index));
    this.syncBoardToPly(this.currentPlyIndex);

    const slider = this.container.querySelector('#ply-slider') as HTMLInputElement;
    if (slider) slider.value = this.currentPlyIndex.toString();

    const plyIndicator = this.container.querySelector('#ply-indicator');
    if (plyIndicator) plyIndicator.textContent = `Ply ${this.currentPlyIndex + 1} / ${this.moves.length}`;
  }

  private render(): void {
    this.container.innerHTML = `
      <!-- MASTHEAD -->
      <header class="masthead shell">
        <div class="masthead-main">
          <a href="#top" class="brand">
            <span>KEPLER-64</span>
            <span class="brand-badge">ROCHE ENGINE</span>
          </a>

          <div class="masthead-actions">
            <button id="nav-export-btn" class="action-secondary masthead-action-btn" title="Export Social GIF">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
              <span>Export GIF</span>
            </button>
            <a href="https://github.com/r-baruah/kepler-64" target="_blank" rel="noreferrer" class="action-primary star-header-btn" title="Star Kepler-64 on GitHub">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              <span>★ Star</span>
            </a>
          </div>
        </div>

        <nav class="nav-links nav-ribbon">
          <a href="#observatory" class="nav-link">Observatory</a>
          <a href="#compendium" class="nav-link">Compendium</a>
          <a href="#mechanism" class="nav-link">How it Works</a>
          <a href="#sandbox" class="nav-link">Laboratory</a>
          <a href="#contributors" class="nav-link">Contributors</a>
        </nav>
      </header>

      <!-- HERO THESIS -->
      <section class="hero shell" id="top">
        <div class="hero-inner">
          <div class="hero-text-block">
            <h1 class="hero-thesis">What if gravity could play chess?</h1>
            <p class="hero-sub">
              Kepler-64 replaces conventional chess heuristics with a differentiable N-body gravitational field.
              Pieces exert mass, warping 2D spacetime—the enemy King gets <strong>tidally disrupted past the Roche limit</strong>.
            </p>
          </div>
          <!-- PRESET SELECTOR -->
          <div class="preset-bar">
            <span class="preset-label">OBSERVATIONS:</span>
            ${PRESET_GAMES.map((g, idx) => `
              <button class="preset-pill ${idx === 0 ? 'active' : ''}" data-preset-id="${g.id}">
                ${g.title}
              </button>
            `).join('')}
          </div>
        </div>
      </section>

      <!-- THE UNIFIED OBSERVATORY CONSOLE -->
      <section class="observatory-console-section shell" id="observatory">
        <div class="observatory-cockpit-grid">
          <!-- LEFT: FUSED BOARD INSTRUMENT + DIRECT PLAYBACK CONTROLS + SPARKLINE -->
          <div class="board-column">
            <div class="board-with-barometer">
              <!-- VERTICAL GRAVITATIONAL BAROMETER -->
              <div class="vertical-barometer" title="Net Spacetime Curvature (White vs Black)">
                <div class="barometer-track">
                  <div id="barometer-fill" class="barometer-fill"></div>
                </div>
                <span id="barometer-val-text" class="barometer-label">+0.0</span>
              </div>

              <!-- BOARD CANVAS -->
              <div class="board-container">
                <canvas id="board-canvas" class="board-canvas"></canvas>
              </div>
            </div>

            <!-- DIRECT PLAYBACK & SCRUBBER CONTROLS (TIED DIRECTLY TO BOARD) -->
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

            <!-- PANORAMIC TRAJECTORY TIMELINE (DIRECTLY BELOW CONTROLS) -->
            <div id="sparkline-container" class="cockpit-sparkline-wrap"></div>

            <!-- HOVER HUD & LAYER TOGGLES STRIP -->
            <div class="board-footer-strip">
              <div class="hover-hud">
                <span id="hover-hud-text">Tap or hover any square to inspect local force & mass</span>
              </div>
              <div class="layer-toggles">
                <button class="toggle-chip active" data-layer="showContours">Contours</button>
                <button class="toggle-chip active" data-layer="showHeatmap">Potential</button>
                <button class="toggle-chip" data-layer="showVectors">Vectors</button>
                <button class="toggle-chip active" data-layer="showTidalStress">Tidal Tensors</button>
                <button class="toggle-chip active" data-layer="showAccretion">Accretion</button>
              </div>
            </div>
          </div>

          <!-- RIGHT: BALANCED TELEMETRY & CANDIDATE DUAL-PANE -->
          <div class="telemetry-column">
            <div class="telemetry-card">
              <div class="telemetry-header">
                <div>
                  <div id="readout-move" class="telemetry-move">—</div>
                  <div id="readout-mover" class="telemetry-mover">White to move</div>
                </div>
                <div id="readout-status" class="telemetry-status status-safe">● TIDAL EQUILIBRIUM</div>
              </div>

              <!-- ROCHE DISRUPTION GAUGE -->
              <div class="roche-gauge-box">
                <div class="gauge-header">
                  <span>King Roche Disruption Index (η)</span>
                  <span id="gauge-eta-text">η = 0.000 / ρ = 0.80</span>
                </div>
                <div class="roche-track">
                  <div id="gauge-eta-fill" class="roche-fill" style="width:15%;"></div>
                </div>
              </div>

              <!-- DUAL PANE: WATERFALL (LEFT) + CANDIDATE MATRIX (RIGHT) -->
              <div class="telemetry-dual-grid">
                <!-- WATERFALL -->
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

                <!-- CANDIDATE MATRIX -->
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

              <!-- NET POSITION SCORE BAR -->
              <div class="net-score-bar">
                <span class="net-score-label">NET POSITION SCORE:</span>
                <span id="total-score-val" class="net-score-val">+0.00 native</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- INTERACTIVE RESEARCH COMPENDIUM (SPLIT-VIEW NOTEBOOK) -->
      <section id="compendium-section" style="padding:var(--space-section) 0; background:#f8fafc; border-top:var(--rule-thin) solid var(--color-rule); border-bottom:var(--rule-thin) solid var(--color-rule);">
        <div id="field-guide-mount"></div>
      </section>

      <!-- ULTRAMARINE MECHANISM PLATE -->
      <section class="mechanism-plate" id="mechanism">
        <div class="shell">
          <div class="badge-tag" style="background:rgba(255,255,255,0.15); color:#fdba74;">ANALYTICAL ENGINE</div>
          <h2>The physics is the evaluator.</h2>
          <p style="max-width:65ch; margin-top:var(--space-sm); opacity:0.9;">
            Kepler-64 evaluates every legal move through an explicit gravitational and tidal tensor calculation.
            Every constant of this universe is a learnable leaf optimized through JAX gradient descent.
          </p>

          <div class="equation-box">
            <div style="color:#fdba74; font-weight:700; margin-bottom:8px;">1. PLUMMER POTENTIAL FIELD</div>
            <div>${latex('\\Phi(p) = -G \\sum_{j=1}^{64} \\frac{|m_j| \\cdot \\sigma(c - \\lVert p - r_j \\rVert)}{\\sqrt{\\lVert p - r_j \\rVert^2 + \\varepsilon^2}}', true)}</div>
            
            <div style="color:#fdba74; font-weight:700; margin-top:20px; margin-bottom:8px;">2. TIDAL TENSOR (HESSIAN) & LINE OF FAILURE</div>
            <div>${latex('\\mathbf{A} = \\nabla\\nabla\\Phi\\big|_{\\text{King}} = \\begin{pmatrix} \\Phi_{xx} & \\Phi_{xy} \\\\ \\Phi_{yx} & \\Phi_{yy} \\end{pmatrix} \\quad \\implies \\quad \\lambda_1, \\lambda_2, \\; \\vec{v}_1', true)}</div>

            <div style="color:#fdba74; font-weight:700; margin-top:20px; margin-bottom:8px;">3. ROCHE DISRUPTION WIN CRITERION</div>
            <div>${latex('\\eta = \\frac{R_g^3 \\cdot \\lambda_1}{m_{\\text{ref}}^2} \\quad \\implies \\quad \\text{King collapses when } \\eta > \\rho_{\\text{roche}}', true)}</div>
          </div>
        </div>
      </section>

      <!-- UNIVERSE SANDBOX & SLIDERS -->
      <section class="shell" id="sandbox" style="padding-bottom:var(--space-section);">
        <div style="margin-bottom:var(--space-lg);">
          <span class="badge-tag">INTERACTIVE LABORATORY</span>
          <h2>Universal Physics Laboratory</h2>
          <p style="color:var(--color-muted); max-width:65ch;">
            Adjust the fundamental physical laws of the chess universe in real time and watch the gravitational topography and tidal tension warp instantly.
          </p>
        </div>

        <div class="sandbox-slider-grid">
          <div class="slider-group">
            <div class="slider-header">
              <span>Gravitational Constant (G)</span>
              <span id="slider-g-val">${this.config.G.toFixed(2)}</span>
            </div>
            <input id="slider-g" type="range" min="0.1" max="4.0" step="0.05" value="${this.config.G}" class="scrubber-slider" />
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Plummer Softening (ε)</span>
              <span id="slider-eps-val">${this.config.eps.toFixed(2)}</span>
            </div>
            <input id="slider-eps" type="range" min="0.1" max="2.5" step="0.05" value="${this.config.eps}" class="scrubber-slider" />
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Speed of Light Gate (c)</span>
              <span id="slider-c-val">${this.config.c.toFixed(2)} sq/ply</span>
            </div>
            <input id="slider-c" type="range" min="1.0" max="10.0" step="0.1" value="${this.config.c}" class="scrubber-slider" />
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Roche Critical Limit (ρ)</span>
              <span id="slider-roche-val">${this.config.roche.toFixed(2)}</span>
            </div>
            <input id="slider-roche" type="range" min="0.1" max="2.0" step="0.05" value="${this.config.roche}" class="scrubber-slider" />
          </div>
        </div>
      </section>

      <!-- CONTRIBUTORS & ATTRIBUTION SECTION -->
      <div id="contributors-mount"></div>

      <!-- FOOTER -->
      <footer style="border-top:var(--rule-thin) solid var(--color-ink); padding:var(--space-xl) 0; background:var(--color-white);">
        <div class="shell footer-inner">
          <div style="font-family:var(--font-mono); font-size:0.82rem; color:var(--color-muted);">
            Kepler-64 © 2026 · Created by <strong>Ripuranjan Baruah</strong> · Open-Source Research
          </div>
          <div class="footer-links">
            <a href="https://github.com/r-baruah/kepler-64" target="_blank" rel="noreferrer" class="action-secondary" style="padding:5px 10px; font-size:0.75rem;">GitHub Repository</a>
            <a href="https://github.com/r-baruah/kepler-64/blob/main/REFERENCES.md" target="_blank" rel="noreferrer" class="action-secondary" style="padding:5px 10px; font-size:0.75rem;">References</a>
            <a href="https://github.com/r-baruah/kepler-64/issues" target="_blank" rel="noreferrer" class="action-secondary" style="padding:5px 10px; font-size:0.75rem;">Report Issue</a>
          </div>
        </div>
      </footer>
    `;

    this.attachEvents();
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
        if (this.canvasRenderer && layer) {
          this.canvasRenderer.setLayers({ [layer]: isActive });
        }
      });
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
          this.updateSparklineData();
        }
      });
    };

    bindSlider('g', 'G');
    bindSlider('eps', 'eps');
    bindSlider('c', 'c', ' sq/ply');
    bindSlider('roche', 'roche');

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
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
