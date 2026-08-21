/**
 * Kepler-64 Observatory GIF & Clip Studio
 * Generates high-resolution Observatory HUD GIFs with live board, telemetry and trajectory.
 */

// @ts-ignore
import gifshot from 'gifshot';
import { KeplerBoard } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { DEFAULT_CONSTANTS } from '../core/constants';
import { BannerRenderer } from '../render/BannerRenderer';
import type { TrajectoryPoint, BannerRenderOptions } from '../render/BannerRenderer';
import { evaluatePosition } from '../core/evaluate';
import { PRESET_GAMES } from '../core/presets';
import type { PresetGame } from '../core/presets';
import { Chess } from 'chess.js';

export class ExportModal {
  private container: HTMLElement;
  private isRendering = false;

  // Customization state
  private activeConfig: ConstantsConfig;
  private availableGames: PresetGame[] = [...PRESET_GAMES];
  private selectedGame: PresetGame;
  private selectedConfig: ConstantsConfig;
  private segmentMode: 'highlight' | 'full' | 'custom' = 'highlight';
  private selectedSpeed: number = 0.8;
  private selectedLayout: 'wide' | 'compact' = 'wide';
  private startPly: number = 1;
  private endPly: number = 24;
  private maxPlies: number = 24;

  // 8 Observatory Visual Layers
  private showHeatmap: boolean = true;
  private showContours: boolean = true;
  private showVectors: boolean = false;
  private showTidalStress: boolean = true;
  private showAccretion: boolean = true;
  private showWavefronts: boolean = false;
  private showLorentz: boolean = false;
  private showLagrange: boolean = false;

  constructor(game: PresetGame, config: ConstantsConfig) {
    this.activeConfig = { ...config };
    this.selectedGame = game;
    this.selectedConfig = { ...config };

    if (!this.availableGames.some((g) => g.id === game.id || g.pgn === game.pgn)) {
      this.availableGames.unshift(game);
    }

    this.container = document.createElement('div');
    this.container.className = 'modal-backdrop';
    this.container.style.display = 'none';
    document.body.appendChild(this.container);

    this.renderModalStructure();
  }

  public open(currentGame: PresetGame, currentConfig: ConstantsConfig): void {
    this.activeConfig = { ...currentConfig };
    this.selectedConfig = { ...currentConfig };

    // Register active observation game if not already present
    if (!this.availableGames.some((g) => g.id === currentGame.id || g.pgn === currentGame.pgn)) {
      this.availableGames.unshift(currentGame);
    }
    this.selectedGame = currentGame;

    this.container.style.display = 'flex';
    this.populateGameMenu();
    this.refreshGameData();
  }

  public close(): void {
    this.closeAllDropdowns();
    this.container.style.display = 'none';
  }

  private closeAllDropdowns(): void {
    this.container.querySelectorAll('.custom-select').forEach((el) => {
      el.classList.remove('open');
    });
  }

  private esc(value: string): string {
    return value.replace(/[&<>"']/g, (ch) => {
      switch (ch) {
        case '&': return '&amp;';
        case '<': return '&lt;';
        case '>': return '&gt;';
        case '"': return '&quot;';
        case "'": return '&#39;';
        default: return ch;
      }
    });
  }

  private populateGameMenu(): void {
    const menu = this.container.querySelector('#cs-game-menu') as HTMLElement;
    const label = this.container.querySelector('#cs-game-label') as HTMLElement;
    if (!menu || !label) return;

    let html = '';
    this.availableGames.forEach((g) => {
      const isPreset = PRESET_GAMES.some((pg) => pg.id === g.id);
      const badge = isPreset ? '[Preset]' : '[Custom]';
      const cleanTitle = g.title.length > 32 ? g.title.slice(0, 30) + '…' : g.title;
      const isSelected = g.id === this.selectedGame.id;
      html += `
        <div class="custom-select-option ${isSelected ? 'selected' : ''}" data-value="${g.id}">
          <span class="opt-badge">${badge}</span>
          <span class="opt-text">${this.esc(cleanTitle)}</span>
        </div>
      `;
    });

    html += `
      <div class="custom-select-option" data-value="__custom_import__" style="color:var(--color-plate); font-weight:700; border-top:1px dashed var(--color-rule);">
        <span class="opt-text">+ Import / Paste Custom PGN...</span>
      </div>
    `;

    menu.innerHTML = html;

    const isCurrentPreset = PRESET_GAMES.some((pg) => pg.id === this.selectedGame.id);
    const badge = isCurrentPreset ? '[Preset]' : '[Custom]';
    const cleanTitle = this.selectedGame.title.length > 32 ? this.selectedGame.title.slice(0, 30) + '…' : this.selectedGame.title;
    label.textContent = `${badge} ${cleanTitle}`;

    // Re-bind option clicks for game dropdown
    menu.querySelectorAll('.custom-select-option').forEach((opt) => {
      opt.addEventListener('click', (e) => {
        e.stopPropagation();
        const val = opt.getAttribute('data-value');
        const pgnBox = this.container.querySelector('#export-pgn-box') as HTMLElement;

        if (val === '__custom_import__') {
          if (pgnBox) pgnBox.style.display = 'block';
          label.textContent = '+ Import Custom PGN';
        } else if (val) {
          if (pgnBox) pgnBox.style.display = 'none';
          const found = this.availableGames.find((g) => g.id === val);
          if (found) {
            this.selectedGame = found;
            this.populateGameMenu();
            this.refreshGameData();
          }
        }
        this.closeAllDropdowns();
      });
    });
  }

  private refreshGameData(): void {
    const chess = new Chess();
    try {
      chess.loadPgn(this.selectedGame.pgn);
    } catch {
      // fallback
    }
    const moves = chess.history();
    this.maxPlies = Math.max(1, moves.length);

    if (this.segmentMode === 'highlight') {
      this.startPly = 1;
      this.endPly = Math.min(this.maxPlies, 24);
    } else if (this.segmentMode === 'full') {
      this.startPly = 1;
      this.endPly = this.maxPlies;
    }

    this.updateRangeInputs();
    this.updateSummaryText();
    this.updatePreview();
  }

  private updateRangeInputs(): void {
    const startInput = this.container.querySelector('#export-start-ply') as HTMLInputElement;
    const endInput = this.container.querySelector('#export-end-ply') as HTMLInputElement;

    if (startInput && endInput) {
      startInput.max = String(this.maxPlies);
      endInput.max = String(this.maxPlies);
      startInput.value = String(this.startPly);
      endInput.value = String(this.endPly);
    }
  }

  private updateSummaryText(): void {
    const summaryEl = this.container.querySelector('#export-summary-text') as HTMLElement;
    if (!summaryEl) return;

    const frameCount = Math.max(1, this.endPly - this.startPly + 1);
    const totalSec = (frameCount * this.selectedSpeed).toFixed(1);

    summaryEl.innerHTML = `
      <span><strong>Target:</strong> ${this.esc(this.selectedGame.title)}</span>
      <span><strong>Plies:</strong> ${this.startPly}–${this.endPly} (${frameCount} frames · ~${totalSec}s)</span>
    `;
  }

  private renderModalStructure(): void {
    this.container.innerHTML = `
      <div class="modal-content">
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:var(--space-md); border-bottom:var(--rule-thin) solid var(--color-rule); padding-bottom:var(--space-sm); flex-wrap:wrap; gap:8px;">
          <div>
            <span class="badge-tag">OBSERVATORY RECORDING STUDIO</span>
            <h3 style="font-size:1.30rem; margin:2px 0 0;">Export Observatory GIF</h3>
          </div>
          <button id="modal-close-btn" class="action-secondary" style="padding:4px 8px;">✕ Close</button>
        </div>

        <!-- 1. Custom Contained Dropdown: Target Match -->
        <div class="studio-field">
          <span class="studio-label">Target Observation Match</span>
          <div class="custom-select" id="cs-game">
            <button type="button" class="custom-select-trigger" id="cs-game-trigger">
              <span class="custom-select-label" id="cs-game-label">[Preset] Kepler-64 Autonomous Clash</span>
              <span class="custom-select-arrow">▼</span>
            </button>
            <div class="custom-select-menu" id="cs-game-menu"></div>
          </div>
        </div>

        <!-- Collapsible Custom PGN Import Box -->
        <div id="export-pgn-box" style="display:none; background:var(--color-surface-2); border:var(--rule-thin) solid var(--color-rule); padding:10px; margin-bottom:12px;">
          <span class="studio-label" style="margin-bottom:4px;">Paste Custom PGN for Analysis &amp; GIF Export</span>
          <input id="export-custom-title" type="text" placeholder="Game Title (e.g., Player vs Stockfish)" style="width:100%; box-sizing:border-box; padding:6px; font-family:var(--font-mono); font-size:0.78rem; border:var(--rule-thin) solid var(--color-ink); margin-bottom:6px;" />
          <textarea id="export-custom-pgn" rows="3" placeholder="1. e4 e5 2. Nf3 Nc6 3. Bb5..." style="width:100%; box-sizing:border-box; padding:6px; font-family:var(--font-mono); font-size:0.75rem; border:var(--rule-thin) solid var(--color-ink); margin-bottom:6px; resize:vertical;"></textarea>
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
            <span id="export-pgn-msg" style="font-family:var(--font-mono); font-size:0.72rem; color:var(--color-accent);"></span>
            <button id="export-load-pgn-btn" class="action-secondary" style="padding:4px 10px; font-size:0.78rem;">Load &amp; Select Game</button>
          </div>
        </div>

        <!-- 2. Dual Column Control Grid: Segment & Speed -->
        <div class="studio-grid-2" style="margin-bottom:12px;">
          <div class="studio-field" style="margin-bottom:0;">
            <span class="studio-label">Sequence Segment</span>
            <div class="custom-select" id="cs-segment">
              <button type="button" class="custom-select-trigger" id="cs-segment-trigger">
                <span class="custom-select-label" id="cs-segment-label">Highlight Arc (24 plies)</span>
                <span class="custom-select-arrow">▼</span>
              </button>
              <div class="custom-select-menu" id="cs-segment-menu">
                <div class="custom-select-option selected" data-value="highlight">
                  <span class="opt-text">Highlight Arc (24 plies)</span>
                </div>
                <div class="custom-select-option" data-value="full">
                  <span class="opt-text">Full Match (Complete)</span>
                </div>
                <div class="custom-select-option" data-value="custom">
                  <span class="opt-text">Custom Move Range...</span>
                </div>
              </div>
            </div>
          </div>

          <div class="studio-field" style="margin-bottom:0;">
            <span class="studio-label">Move Speed / Interval</span>
            <div class="custom-select" id="cs-speed">
              <button type="button" class="custom-select-trigger" id="cs-speed-trigger">
                <span class="custom-select-label" id="cs-speed-label">0.8s / move (Standard)</span>
                <span class="custom-select-arrow">▼</span>
              </button>
              <div class="custom-select-menu" id="cs-speed-menu">
                <div class="custom-select-option selected" data-value="0.8">
                  <span class="opt-text">0.8s / move (Standard)</span>
                </div>
                <div class="custom-select-option" data-value="0.5">
                  <span class="opt-text">0.5s / move (Fast Playback)</span>
                </div>
                <div class="custom-select-option" data-value="1.2">
                  <span class="opt-text">1.2s / move (Deep Analysis)</span>
                </div>
                <div class="custom-select-option" data-value="0.3">
                  <span class="opt-text">0.3s / move (Rapid Scan)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Custom Range Steppers (if custom segment selected) -->
        <div id="custom-range-row" class="export-range-inputs" style="margin-bottom:12px; display:none;">
          <div class="studio-field" style="margin-bottom:0;">
            <span class="studio-label">Start Ply</span>
            <input id="export-start-ply" type="number" min="1" max="80" value="1" style="padding:7px 10px; font-family:var(--font-mono); font-size:0.80rem; border:var(--rule-thin) solid var(--color-ink); width:100%; box-sizing:border-box;" />
          </div>
          <div class="studio-field" style="margin-bottom:0;">
            <span class="studio-label">End Ply</span>
            <input id="export-end-ply" type="number" min="1" max="80" value="24" style="padding:7px 10px; font-family:var(--font-mono); font-size:0.80rem; border:var(--rule-thin) solid var(--color-ink); width:100%; box-sizing:border-box;" />
          </div>
        </div>

        <!-- 3. Dual Column Control Grid: Physics & Canvas Format -->
        <div class="studio-grid-2" style="margin-bottom:12px;">
          <div class="studio-field" style="margin-bottom:0;">
            <span class="studio-label">Physics Profile</span>
            <div class="custom-select" id="cs-physics">
              <button type="button" class="custom-select-trigger" id="cs-physics-trigger">
                <span class="custom-select-label" id="cs-physics-label">Active Settings</span>
                <span class="custom-select-arrow">▼</span>
              </button>
              <div class="custom-select-menu" id="cs-physics-menu">
                <div class="custom-select-option selected" data-value="active">
                  <span class="opt-text">Active Settings</span>
                </div>
                <div class="custom-select-option" data-value="trained">
                  <span class="opt-text">Standard Trained (G=1.0)</span>
                </div>
                <div class="custom-select-option" data-value="heavy">
                  <span class="opt-text">High-Gravity (G=1.8)</span>
                </div>
                <div class="custom-select-option" data-value="relativistic">
                  <span class="opt-text">Relativistic (c=2.2)</span>
                </div>
              </div>
            </div>
          </div>

          <div class="studio-field" style="margin-bottom:0;">
            <span class="studio-label">Canvas Format</span>
            <div class="custom-select" id="cs-layout">
              <button type="button" class="custom-select-trigger" id="cs-layout-trigger">
                <span class="custom-select-label" id="cs-layout-label">Widescreen HD (840×520)</span>
                <span class="custom-select-arrow">▼</span>
              </button>
              <div class="custom-select-menu" id="cs-layout-menu">
                <div class="custom-select-option selected" data-value="wide">
                  <span class="opt-text">Widescreen HD (840×520)</span>
                </div>
                <div class="custom-select-option" data-value="compact">
                  <span class="opt-text">Compact HUD (700×430)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 4. Complete 8 Visual Layer Toggles -->
        <div class="studio-field">
          <span class="studio-label">Render Overlays (Observatory Suite)</span>
          <div class="export-layer-toggles">
            <button type="button" class="export-layer-chip active" id="chip-heatmap">
              <span>● Potential</span>
            </button>
            <button type="button" class="export-layer-chip active" id="chip-contours">
              <span>● Contours</span>
            </button>
            <button type="button" class="export-layer-chip" id="chip-vectors">
              <span>● Vectors</span>
            </button>
            <button type="button" class="export-layer-chip active" id="chip-tensors">
              <span>● Tidal</span>
            </button>
            <button type="button" class="export-layer-chip active" id="chip-accretion">
              <span>● Accretion</span>
            </button>
            <button type="button" class="export-layer-chip" id="chip-wavefronts">
              <span>● Light Cone</span>
            </button>
            <button type="button" class="export-layer-chip" id="chip-lorentz">
              <span>● Kinetic (γ)</span>
            </button>
            <button type="button" class="export-layer-chip" id="chip-lagrange">
              <span>● Lagrange</span>
            </button>
          </div>
        </div>

        <!-- Live Configuration Summary Banner -->
        <div id="export-summary-text" class="export-stat-banner">
          <span><strong>Target:</strong> ${this.esc(this.selectedGame.title)}</span>
          <span><strong>Plies:</strong> 1–24 (24 frames)</span>
        </div>

        <div id="export-status" style="margin-bottom:var(--space-md); font-family:var(--font-mono); font-size:0.8rem; display:none;">
          <div style="background:#e2e8f0; height:6px; margin-bottom:6px; overflow:hidden;">
            <div id="export-progress-bar" style="width:0%; height:100%; background:var(--color-accent); transition:width 0.2s ease;"></div>
          </div>
          <span id="export-status-text">Preloading piece assets...</span>
        </div>

        <div id="export-result-container" style="margin-bottom:var(--space-md); text-align:center; display:none;">
          <img id="export-result-img" style="max-width:100%; height:auto; object-fit:contain; border:var(--rule-thin) solid var(--color-ink); max-height:260px;" alt="Exported Kepler-64 Observatory GIF preview" />
        </div>

        <div class="modal-actions">
          <button id="render-btn" class="action-primary" style="width:100%; justify-content:center; min-height:44px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            Generate &amp; Download GIF
          </button>
        </div>
      </div>
    `;

    this.bindEvents();
  }

  private setupCustomDropdown(
    selectId: string,
    onSelect: (val: string, label: string) => void
  ): void {
    const root = this.container.querySelector(`#${selectId}`) as HTMLElement;
    if (!root) return;

    const trigger = root.querySelector('.custom-select-trigger') as HTMLElement;
    const label = root.querySelector('.custom-select-label') as HTMLElement;
    const menu = root.querySelector('.custom-select-menu') as HTMLElement;

    trigger?.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = root.classList.contains('open');
      this.closeAllDropdowns();
      if (!isOpen) {
        root.classList.add('open');
      }
    });

    menu?.querySelectorAll('.custom-select-option').forEach((opt) => {
      opt.addEventListener('click', (e) => {
        e.stopPropagation();
        const val = opt.getAttribute('data-value') || '';
        const optText = opt.querySelector('.opt-text')?.textContent || opt.textContent || '';
        menu.querySelectorAll('.custom-select-option').forEach((o) => o.classList.remove('selected'));
        opt.classList.add('selected');
        if (label) label.textContent = optText.trim();
        root.classList.remove('open');
        onSelect(val, optText.trim());
      });
    });
  }

  private bindEvents(): void {
    this.container.querySelector('#modal-close-btn')?.addEventListener('click', () => this.close());
    this.container.addEventListener('click', (e) => {
      if (e.target === this.container) {
        this.close();
      } else {
        this.closeAllDropdowns();
      }
    });

    // Setup Target Game Trigger click
    const gameRoot = this.container.querySelector('#cs-game') as HTMLElement;
    const gameTrigger = this.container.querySelector('#cs-game-trigger') as HTMLElement;
    gameTrigger?.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = gameRoot.classList.contains('open');
      this.closeAllDropdowns();
      if (!isOpen) gameRoot.classList.add('open');
    });

    // Custom PGN Loader inside Export Modal
    const loadPgnBtn = this.container.querySelector('#export-load-pgn-btn') as HTMLButtonElement;
    const pgnInput = this.container.querySelector('#export-custom-pgn') as HTMLTextAreaElement;
    const titleInput = this.container.querySelector('#export-custom-title') as HTMLInputElement;
    const pgnMsg = this.container.querySelector('#export-pgn-msg') as HTMLElement;
    const pgnBox = this.container.querySelector('#export-pgn-box') as HTMLElement;

    loadPgnBtn?.addEventListener('click', () => {
      const pgn = pgnInput?.value?.trim();
      if (!pgn) {
        if (pgnMsg) pgnMsg.textContent = 'Please enter valid PGN moves.';
        return;
      }

      const chess = new Chess();
      try {
        chess.loadPgn(pgn);
        const moves = chess.history();
        if (moves.length === 0) {
          if (pgnMsg) pgnMsg.textContent = 'PGN contains no playable moves.';
          return;
        }

        const title = titleInput?.value?.trim() || `Custom Match (${moves.length} plies)`;
        const customGame: PresetGame = {
          id: `custom_${Date.now()}`,
          title,
          subtitle: `Imported PGN · ${moves.length} plies`,
          white: 'White',
          black: 'Black',
          date: new Date().toISOString().slice(0, 10),
          event: 'Observatory Custom Analysis',
          initialFen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          pgn,
          highlightPly: Math.min(moves.length, 12),
        };

        this.availableGames.unshift(customGame);
        this.selectedGame = customGame;
        this.populateGameMenu();
        if (pgnBox) pgnBox.style.display = 'none';
        if (pgnMsg) pgnMsg.textContent = '';
        this.refreshGameData();
      } catch (err) {
        if (pgnMsg) pgnMsg.textContent = `Invalid PGN format: ${String(err)}`;
      }
    });

    // 1. Sequence Segment Dropdown
    const customRow = this.container.querySelector('#custom-range-row') as HTMLElement;
    this.setupCustomDropdown('cs-segment', (val) => {
      this.segmentMode = val as 'highlight' | 'full' | 'custom';
      if (val === 'highlight') {
        if (customRow) customRow.style.display = 'none';
        this.startPly = 1;
        this.endPly = Math.min(this.maxPlies, 24);
      } else if (val === 'full') {
        if (customRow) customRow.style.display = 'none';
        this.startPly = 1;
        this.endPly = this.maxPlies;
      } else if (val === 'custom') {
        if (customRow) customRow.style.display = 'grid';
      }
      this.updateRangeInputs();
      this.updateSummaryText();
    });

    // 2. Speed Dropdown
    this.setupCustomDropdown('cs-speed', (val) => {
      this.selectedSpeed = parseFloat(val) || 0.8;
      this.updateSummaryText();
    });

    // 3. Physics Preset Dropdown
    this.setupCustomDropdown('cs-physics', (val) => {
      switch (val) {
        case 'trained':
          this.selectedConfig = { ...DEFAULT_CONSTANTS };
          break;
        case 'heavy':
          this.selectedConfig = { ...DEFAULT_CONSTANTS, G: 1.8, eps: 0.35, roche: 0.8, bonus: 350 };
          break;
        case 'relativistic':
          this.selectedConfig = { ...DEFAULT_CONSTANTS, c: 2.2, eps: 0.4, bonus: 400 };
          break;
        case 'active':
        default:
          this.selectedConfig = { ...this.activeConfig };
          break;
      }
    });

    // 4. Layout Dropdown
    this.setupCustomDropdown('cs-layout', (val) => {
      this.selectedLayout = val as 'wide' | 'compact';
    });

    // Ply inputs
    const startInput = this.container.querySelector('#export-start-ply') as HTMLInputElement;
    const endInput = this.container.querySelector('#export-end-ply') as HTMLInputElement;

    startInput?.addEventListener('input', () => {
      let val = parseInt(startInput.value, 10) || 1;
      val = Math.max(1, Math.min(this.maxPlies, val));
      this.startPly = val;
      if (this.startPly > this.endPly) {
        this.endPly = this.startPly;
        if (endInput) endInput.value = String(this.endPly);
      }
      this.updateSummaryText();
    });

    endInput?.addEventListener('input', () => {
      let val = parseInt(endInput.value, 10) || 1;
      val = Math.max(1, Math.min(this.maxPlies, val));
      this.endPly = val;
      if (this.endPly < this.startPly) {
        this.startPly = this.endPly;
        if (startInput) startInput.value = String(this.startPly);
      }
      this.updateSummaryText();
    });

    // Layer Toggle Chips for all 8 layers
    const bindChip = (
      id: string,
      prop:
        | 'showHeatmap'
        | 'showContours'
        | 'showVectors'
        | 'showTidalStress'
        | 'showAccretion'
        | 'showWavefronts'
        | 'showLorentz'
        | 'showLagrange'
    ) => {
      const chip = this.container.querySelector(id) as HTMLElement;
      chip?.addEventListener('click', () => {
        this[prop] = !this[prop];
        chip.classList.toggle('active', this[prop]);
      });
    };

    bindChip('#chip-heatmap', 'showHeatmap');
    bindChip('#chip-contours', 'showContours');
    bindChip('#chip-vectors', 'showVectors');
    bindChip('#chip-tensors', 'showTidalStress');
    bindChip('#chip-accretion', 'showAccretion');
    bindChip('#chip-wavefronts', 'showWavefronts');
    bindChip('#chip-lorentz', 'showLorentz');
    bindChip('#chip-lagrange', 'showLagrange');

    // Render Button
    this.container.querySelector('#render-btn')?.addEventListener('click', () => this.executeExport());
  }

  private updatePreview(): void {
    const statusBox = this.container.querySelector('#export-status') as HTMLElement;
    const resultBox = this.container.querySelector('#export-result-container') as HTMLElement;
    if (statusBox) statusBox.style.display = 'none';
    if (resultBox) resultBox.style.display = 'none';
  }

  private async executeExport(): Promise<void> {
    if (this.isRendering) return;
    this.isRendering = true;

    const statusBox = this.container.querySelector('#export-status') as HTMLElement;
    const progressBar = this.container.querySelector('#export-progress-bar') as HTMLElement;
    const statusText = this.container.querySelector('#export-status-text') as HTMLElement;
    const resultBox = this.container.querySelector('#export-result-container') as HTMLElement;
    const resultImg = this.container.querySelector('#export-result-img') as HTMLImageElement;
    const renderBtn = this.container.querySelector('#render-btn') as HTMLButtonElement;

    statusBox.style.display = 'block';
    renderBtn.disabled = true;
    statusText.textContent = 'Preloading vector chess pieces...';

    try {
      // 1. Ensure all piece SVG assets are 100% preloaded in memory
      await BannerRenderer.loadPieceImages();

      const chess = new Chess();
      chess.loadPgn(this.selectedGame.pgn);
      const moves = chess.history({ verbose: true });

      // Precompute game trajectory points
      const trajectoryBoard = new KeplerBoard();
      const trajectoryChess = new Chess();
      const trajectoryPoints: TrajectoryPoint[] = [];

      for (let i = 0; i < moves.length; i++) {
        const m = moves[i];
        trajectoryChess.move(m);
        trajectoryBoard.loadFen(trajectoryChess.fen());
        const evalRes = evaluatePosition(trajectoryBoard, this.selectedConfig);
        trajectoryPoints.push({
          ply: i,
          score: evalRes.totalScoreWhite,
          moveSan: m.san,
        });
      }

      const isHD = this.selectedLayout === 'wide';
      const canvasWidth = isHD ? 840 : 700;
      const canvasHeight = isHD ? 520 : 430;

      const offCanvas = document.createElement('canvas');
      offCanvas.width = canvasWidth;
      offCanvas.height = canvasHeight;

      const renderOptions: BannerRenderOptions = {
        showHeatmap: this.showHeatmap,
        showContours: this.showContours,
        showVectors: this.showVectors,
        showTidalStress: this.showTidalStress,
        showAccretion: this.showAccretion,
        showWavefronts: this.showWavefronts,
        showLorentz: this.showLorentz,
        showLagrange: this.showLagrange,
      };

      const replayBoard = new KeplerBoard();
      const replayChess = new Chess();

      // Determine ply slice [startIdx, endIdx] (0-indexed)
      const startIdx = Math.max(0, this.startPly - 1);
      const endIdx = Math.min(moves.length - 1, this.endPly - 1);
      const frameCount = Math.max(1, endIdx - startIdx + 1);

      // Fast-forward replayChess and replayBoard up to startIdx
      for (let i = 0; i < startIdx; i++) {
        replayChess.move(moves[i]);
      }
      replayBoard.loadFen(replayChess.fen());

      const frameUrls: string[] = [];

      for (let i = startIdx; i <= endIdx; i++) {
        const move = moves[i];
        replayChess.move(move);
        replayBoard.loadFen(replayChess.fen());

        const lastMoveObj = {
          from: (move.from.charCodeAt(1) - 49) * 8 + (move.from.charCodeAt(0) - 97),
          to: (move.to.charCodeAt(1) - 49) * 8 + (move.to.charCodeAt(0) - 97),
        };

        const moveSan = `${Math.floor(i / 2) + 1}. ${move.color === 'b' ? '...' : ''}${move.san}`;

        BannerRenderer.renderFrame(
          offCanvas,
          replayBoard,
          this.selectedConfig,
          this.selectedGame.title,
          moveSan,
          i,
          moves.length,
          trajectoryPoints,
          lastMoveObj,
          renderOptions
        );

        frameUrls.push(offCanvas.toDataURL('image/png'));

        const currentFrameIndex = i - startIdx + 1;
        const pct = Math.round((currentFrameIndex / frameCount) * 45);
        progressBar.style.width = `${pct}%`;
        statusText.textContent = `Rendered composite frame ${currentFrameIndex} of ${frameCount}...`;
      }

      statusText.textContent = 'Encoding animated GIF with gifshot...';

      gifshot.createGIF(
        {
          images: frameUrls,
          interval: this.selectedSpeed,
          gifWidth: canvasWidth,
          gifHeight: canvasHeight,
          progressCallback: (captureProgress: number) => {
            const pct = 45 + Math.round(captureProgress * 55);
            progressBar.style.width = `${pct}%`;
          },
        },
        (obj: { error: boolean; image: string }) => {
          if (!obj.error) {
            resultImg.src = obj.image;
            resultBox.style.display = 'block';
            statusText.textContent = '✓ High-fidelity GIF generated successfully!';

            // Download
            const a = document.createElement('a');
            a.href = obj.image;
            a.download = `kepler64_${this.selectedGame.id}_plies_${this.startPly}-${this.endPly}.gif`;
            a.click();
          } else {
            statusText.textContent = 'Error assembling GIF.';
          }
          this.isRendering = false;
          renderBtn.disabled = false;
        }
      );
    } catch (err) {
      console.error(err);
      statusText.textContent = 'Render failed: ' + String(err);
      this.isRendering = false;
      renderBtn.disabled = false;
    }
  }
}
