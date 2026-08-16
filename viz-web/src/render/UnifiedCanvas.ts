/**
 * Kepler-64 Unified Canvas Renderer
 * Renders the fused Chessboard + Topographic Gravity Field + Force Vectors + Tidal Tensors
 */

import { KeplerBoard } from '../core/board';
import type { Piece } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { DEFAULT_CONSTANTS } from '../core/constants';
import { potentialOnGrid, forceField, DIST_64 } from '../core/gravity';
import { evaluatePosition } from '../core/evaluate';
import type { ScoreBreakdown } from '../core/evaluate';
import { generateContourLines } from './ContourRenderer';
import { drawTidalStressEllipse } from './TidalRenderer';
import { drawAccretionHalos, drawCaptureStream } from './AccretionRenderer';

export interface RenderLayers {
  showHeatmap: boolean;
  showContours: boolean;
  showVectors: boolean;
  showTidalStress: boolean;
  showAccretion: boolean;
  showWavefronts: boolean;
  showLorentz: boolean;
  showLagrange: boolean;
}

export interface SquareHoverInfo {
  square: number;
  fileChar: string;
  rankNum: number;
  piece: Piece | null;
  mass: number;
  potential: number;
  forceMag: number;
}

export class UnifiedCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private board: KeplerBoard;
  private config: ConstantsConfig;
  private layers: RenderLayers = {
    showHeatmap: true,
    showContours: true,
    showVectors: false,
    showTidalStress: true,
    showAccretion: true,
    showWavefronts: false,
    showLorentz: false,
    showLagrange: false,
  };

  private pieceImages: Map<string, HTMLImageElement> = new Map();
  private isLoaded = false;
  private lastMove: { from: number; to: number } | null = null;
  private accretionExcess: Record<number, number> = {};
  private captureStream: { fromSq: number; toSq: number; startTime: number } | null = null;
  private animFrame: number | null = null;
  private pulseTimer: number | null = null;

  // Memoized expensive-layer cache (heatmap + contours, vectors, evaluation).
  private cacheKey: string | null = null;
  private heatmapCanvas: HTMLCanvasElement | null = null;
  private cachedForces: { fx: Float32Array; fy: Float32Array } | null = null;
  private cachedBreakdown: ScoreBreakdown | null = null;

  // Hover state
  private hoveredSquare: number | null = null;
  private onHoverCallback?: (info: SquareHoverInfo | null) => void;

  // Selection & Tap-to-move state
  private selectedSquare: number | null = null;

  // Drag-and-drop state
  private isDragging = false;
  private dragPiece: Piece | null = null;
  private dragFromSq: number | null = null;
  private dragStartX = 0;
  private dragStartY = 0;
  private dragLastClientX = 0;
  private dragLastClientY = 0;
  private dragCanvasX = 0;
  private dragCanvasY = 0;
  private onMoveCallback?: (fromSq: number, toSq: number) => void;
  private onEvaluateCallback?: (breakdown: ScoreBreakdown) => void;
  private orientation: 'w' | 'b' = 'w';

  // Bound listener references so destroy() can remove them.
  private boundCanvasMouseDown = this.handlePointerDown.bind(this);
  private boundCanvasMouseMove = this.handleMouseMove.bind(this);
  private boundCanvasMouseLeave = this.handleMouseLeave.bind(this);
  private boundWindowMouseMove = this.handleWindowMouseMove.bind(this);
  private boundWindowMouseUp = this.handlePointerUp.bind(this);
  private boundTouchStart = (e: TouchEvent): void => {
    if (e.touches.length === 1) {
      const touch = e.touches[0];
      this.startDrag(touch.clientX, touch.clientY);
    }
  };
  private boundTouchMove = (e: TouchEvent): void => {
    if (this.isDragging && e.touches.length === 1) {
      if (e.cancelable) e.preventDefault();
      const touch = e.touches[0];
      this.updateDrag(touch.clientX, touch.clientY);
    }
  };
  private boundTouchEnd = (e: TouchEvent): void => {
    if (this.isDragging) {
      if (e.changedTouches.length > 0) {
        const touch = e.changedTouches[0];
        this.endDrag(touch.clientX, touch.clientY);
      } else {
        this.endDrag();
      }
    }
  };

  constructor(
    canvas: HTMLCanvasElement,
    board: KeplerBoard,
    config: ConstantsConfig = DEFAULT_CONSTANTS
  ) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not obtain 2D canvas context');
    this.ctx = ctx;
    this.board = board;
    this.config = config;

    this.preloadPieces();
    this.attachEventListeners();
    this.syncPulseTimer();
  }

  public setBoard(board: KeplerBoard, lastMove: { from: number; to: number } | null = null): void {
    this.board = board;
    this.lastMove = lastMove;
    this.selectedSquare = null;
    this.captureStream = null;
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }
    this.render();
  }

  public setConfig(config: ConstantsConfig): void {
    this.config = config;
    this.render();
  }

  public setLayers(layers: Partial<RenderLayers>): void {
    this.layers = { ...this.layers, ...layers };
    this.syncPulseTimer();
    this.render();
  }

  private syncPulseTimer(): void {
    const shouldPulse =
      this.layers.showAccretion || this.layers.showWavefronts || this.layers.showLorentz;
    if (shouldPulse && this.pulseTimer === null) {
      this.pulseTimer = window.setInterval(() => this.render(), 200);
    } else if (!shouldPulse && this.pulseTimer !== null) {
      window.clearInterval(this.pulseTimer);
      this.pulseTimer = null;
    }
  }

  public setAccretion(excess: Record<number, number>): void {
    this.accretionExcess = excess;
  }

  public setOrientation(color: 'w' | 'b'): void {
    this.orientation = color;
    this.render();
  }

  public destroy(): void {
    if (this.pulseTimer !== null) {
      window.clearInterval(this.pulseTimer);
      this.pulseTimer = null;
    }
    if (this.animFrame !== null) {
      cancelAnimationFrame(this.animFrame);
      this.animFrame = null;
    }

    this.canvas.removeEventListener('mousedown', this.boundCanvasMouseDown);
    this.canvas.removeEventListener('mousemove', this.boundCanvasMouseMove);
    this.canvas.removeEventListener('mouseleave', this.boundCanvasMouseLeave);
    window.removeEventListener('mousemove', this.boundWindowMouseMove);
    window.removeEventListener('mouseup', this.boundWindowMouseUp);
    this.canvas.removeEventListener('touchstart', this.boundTouchStart);
    window.removeEventListener('touchmove', this.boundTouchMove);
    window.removeEventListener('touchend', this.boundTouchEnd);

    this.cacheKey = null;
    this.heatmapCanvas = null;
    this.cachedForces = null;
    this.cachedBreakdown = null;
  }

  private disp(sq: number): number {
    return this.orientation === 'b' ? 63 - sq : sq;
  }

  public playCaptureStream(fromSq: number, toSq: number): void {
    this.captureStream = { fromSq, toSq, startTime: performance.now() };
    if (this.animFrame === null) {
      this.tickAnimation();
    }
  }

  private tickAnimation = (): void => {
    if (!this.captureStream) {
      this.animFrame = null;
      return;
    }
    const t = (performance.now() - this.captureStream.startTime) / 300;
    if (t >= 1) {
      this.captureStream = null;
      this.animFrame = null;
      this.render();
      return;
    }
    this.render();
    this.animFrame = requestAnimationFrame(this.tickAnimation);
  };

  public onMove(cb: (fromSq: number, toSq: number) => void): void {
    this.onMoveCallback = cb;
  }

  public onEvaluate(cb: (breakdown: ScoreBreakdown) => void): void {
    this.onEvaluateCallback = cb;
  }

  public onHover(cb: (info: SquareHoverInfo | null) => void): void {
    this.onHoverCallback = cb;
  }

  private preloadPieces(): void {
    const pieces = [
      'bb', 'bk', 'bn', 'bp', 'bq', 'br',
      'wb', 'wk', 'wn', 'wp', 'wq', 'wr',
    ];
    let loadedCount = 0;

    const base = (import.meta.env.BASE_URL || './').replace(/\/$/, '') + '/';
    pieces.forEach((name) => {
      const img = new Image();
      img.src = `${base}pieces/${name}.svg`;
      img.onload = () => {
        this.pieceImages.set(name, img);
        loadedCount++;
        if (loadedCount === pieces.length) {
          this.isLoaded = true;
          this.render();
        }
      };
      img.onerror = () => {
        console.error(`Failed to load piece image: ${base}pieces/${name}.svg`);
      };
    });
  }

  private attachEventListeners(): void {
    this.canvas.addEventListener('mousedown', this.boundCanvasMouseDown);
    this.canvas.addEventListener('mousemove', this.boundCanvasMouseMove);
    this.canvas.addEventListener('mouseleave', this.boundCanvasMouseLeave);
    window.addEventListener('mousemove', this.boundWindowMouseMove);
    window.addEventListener('mouseup', this.boundWindowMouseUp);

    // Touch support with touch-action lock
    this.canvas.addEventListener('touchstart', this.boundTouchStart, { passive: false });
    window.addEventListener('touchmove', this.boundTouchMove, { passive: false });
    window.addEventListener('touchend', this.boundTouchEnd);
  }

  private getCanvasSquare(clientX: number, clientY: number): number | null {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    if (x < 0 || x > rect.width || y < 0 || y > rect.height) return null;

    const file = Math.floor((x / rect.width) * 8);
    const rank = 7 - Math.floor((y / rect.height) * 8);
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
    return rank * 8 + file;
  }

  private handleMouseMove(e: MouseEvent): void {
    if (this.isDragging) return;
    const sq = this.getCanvasSquare(e.clientX, e.clientY);
    if (sq !== this.hoveredSquare) {
      this.hoveredSquare = sq;
      if (this.onHoverCallback) {
        if (sq !== null) {
          const realSq = this.disp(sq);
          const f = realSq % 8;
          const r = Math.floor(realSq / 8);
          const piece = this.board.squares[realSq];
          const masses = this.board.massVector();
          const forces = forceField(masses, this.config.eps, this.config.G, this.config.c);
          const fMag = Math.sqrt(forces.fx[realSq] * forces.fx[realSq] + forces.fy[realSq] * forces.fy[realSq]);

          this.onHoverCallback({
            square: realSq,
            fileChar: String.fromCharCode(97 + f),
            rankNum: r + 1,
            piece,
            mass: piece ? piece.mass : 0,
            potential: 0,
            forceMag: fMag,
          });
        } else {
          this.onHoverCallback(null);
        }
      }
      this.render();
    }
  }

  private handleMouseLeave(): void {
    this.hoveredSquare = null;
    if (this.onHoverCallback) this.onHoverCallback(null);
    this.render();
  }

  private handlePointerDown(e: MouseEvent): void {
    this.startDrag(e.clientX, e.clientY);
  }

  private startDrag(clientX: number, clientY: number): void {
    this.dragStartX = clientX;
    this.dragStartY = clientY;
    this.dragLastClientX = clientX;
    this.dragLastClientY = clientY;

    const rect = this.canvas.getBoundingClientRect();
    this.dragCanvasX = ((clientX - rect.left) / rect.width) * this.canvas.width;
    this.dragCanvasY = ((clientY - rect.top) / rect.height) * this.canvas.height;

    const sq = this.getCanvasSquare(clientX, clientY);
    if (sq === null) return;
    const piece = this.board.squares[this.disp(sq)];

    this.isDragging = true;
    this.dragPiece = piece;
    this.dragFromSq = sq;
    this.render();
  }

  private handleWindowMouseMove(e: MouseEvent): void {
    if (!this.isDragging) return;
    this.updateDrag(e.clientX, e.clientY);
  }

  private updateDrag(clientX: number, clientY: number): void {
    this.dragLastClientX = clientX;
    this.dragLastClientY = clientY;

    const rect = this.canvas.getBoundingClientRect();
    this.dragCanvasX = ((clientX - rect.left) / rect.width) * this.canvas.width;
    this.dragCanvasY = ((clientY - rect.top) / rect.height) * this.canvas.height;
    this.hoveredSquare = this.getCanvasSquare(clientX, clientY);
    this.render();
  }

  private handlePointerUp(e: MouseEvent): void {
    if (!this.isDragging) return;
    this.endDrag(e.clientX, e.clientY);
  }

  private endDrag(clientX?: number, clientY?: number): void {
    if (!this.isDragging) return;

    const finalX = clientX !== undefined ? clientX : this.dragLastClientX;
    const finalY = clientY !== undefined ? clientY : this.dragLastClientY;
    const dragDistance = Math.hypot(finalX - this.dragStartX, finalY - this.dragStartY);
    const targetSq = this.getCanvasSquare(finalX, finalY);

    if (dragDistance < 10) {
      // TAP / CLICK INTERACTION (Tap-to-move)
      if (this.selectedSquare === null) {
        if (targetSq !== null && this.board.squares[this.disp(targetSq)]) {
          this.selectedSquare = targetSq;
        }
      } else {
        if (targetSq === this.selectedSquare) {
          // Deselect
          this.selectedSquare = null;
        } else if (targetSq !== null) {
          const fromReal = this.disp(this.selectedSquare);
          const toReal = this.disp(targetSq);
          if (this.onMoveCallback) {
            this.onMoveCallback(fromReal, toReal);
          } else {
            const piece = this.board.squares[fromReal];
            this.board.squares[toReal] = piece;
            this.board.squares[fromReal] = null;
            this.lastMove = { from: fromReal, to: toReal };
          }
          this.selectedSquare = null;
        }
      }
    } else {
      // DRAG & DROP INTERACTION
      if (targetSq !== null && this.dragFromSq !== null && targetSq !== this.dragFromSq && this.dragPiece) {
        const fromReal = this.disp(this.dragFromSq);
        const toReal = this.disp(targetSq);
        if (this.onMoveCallback) {
          this.onMoveCallback(fromReal, toReal);
        } else {
          // Direct local board move
          this.board.squares[toReal] = this.dragPiece;
          this.board.squares[fromReal] = null;
          this.lastMove = { from: fromReal, to: toReal };
        }
        this.selectedSquare = null;
      }
    }

    this.isDragging = false;
    this.dragPiece = null;
    this.dragFromSq = null;
    this.render();
  }

  private computeLagrangePoints(): { x: number; y: number; label: string }[] {
    const masses = this.board.massVector();
    const ranked: { sq: number; m: number }[] = [];
    for (let sq = 0; sq < 64; sq++) {
      if (this.board.squares[sq]) ranked.push({ sq, m: Math.abs(masses[sq]) });
    }
    ranked.sort((a, b) => b.m - a.m);
    if (ranked.length < 2 || ranked[0].m <= 0 || ranked[1].m <= 0) return [];

    const A = ranked[0]; // primary (heavier)
    const B = ranked[1]; // secondary
    const ax = (A.sq % 8) + 0.5;
    const ay = Math.floor(A.sq / 8) + 0.5;
    const bx = (B.sq % 8) + 0.5;
    const by = Math.floor(B.sq / 8) + 0.5;

    const dx = bx - ax;
    const dy = by - ay;
    const d = Math.hypot(dx, dy) || 1;
    const ux = dx / d;
    const uy = dy / d;

    const mu = B.m / (A.m + B.m); // secondary mass fraction
    const s60 = 0.8660254;

    let points: { x: number; y: number; label: string }[];
    if (Math.abs(mu - 0.5) < 1e-6) {
      // Equal dominant masses: the classical collinear approximations are
      // singular; use the symmetric midpoint / symmetric beyond-colinear points.
      points = [
        { x: ax + ux * d * 0.5, y: ay + uy * d * 0.5, label: 'L1' },
        { x: bx + ux * d * 0.55, y: by + uy * d * 0.55, label: 'L2' },
        { x: ax - ux * d * 0.55, y: ay - uy * d * 0.55, label: 'L3' },
        { x: ax + (ux * 0.5 - uy * s60) * d, y: ay + (uy * 0.5 + ux * s60) * d, label: 'L4' },
        { x: ax + (ux * 0.5 + uy * s60) * d, y: ay + (uy * 0.5 - ux * s60) * d, label: 'L5' },
      ];
    } else {
      const alpha = Math.cbrt(mu / 3);
      points = [
        { x: ax + ux * d * (1 - alpha), y: ay + uy * d * (1 - alpha), label: 'L1' },
        { x: bx + ux * d * alpha, y: by + uy * d * alpha, label: 'L2' },
        { x: ax - ux * d * (1 - (7 * mu) / 12), y: ay - uy * d * (1 - (7 * mu) / 12), label: 'L3' },
        { x: ax + (ux * 0.5 - uy * s60) * d, y: ay + (uy * 0.5 + ux * s60) * d, label: 'L4' },
        { x: ax + (ux * 0.5 + uy * s60) * d, y: ay + (uy * 0.5 - ux * s60) * d, label: 'L5' },
      ];
    }

    return points.filter((p) => p.x >= 0 && p.x <= 8 && p.y >= 0 && p.y <= 8);
  }

  private buildCacheKey(width: number, height: number): string {
    const layerFlags = `${this.layers.showHeatmap ? 1 : 0}${this.layers.showContours ? 1 : 0}${this.layers.showVectors ? 1 : 0}`;
    // toFen() does not include massBoost, which changes massVector() for
    // accretion/Lorentz boosts; keep it in the key so equal FENs with
    // different boosts still invalidate the expensive-layer cache.
    const massBoost = Array.from(this.board.massBoost).join(',');
    const dragState =
      this.isDragging && this.dragFromSq !== null
        ? `${this.dragFromSq}:${this.hoveredSquare ?? this.dragFromSq}`
        : '';
    return [
      this.board.toFen(),
      JSON.stringify(this.config),
      layerFlags,
      this.orientation,
      width,
      height,
      massBoost,
      dragState,
    ].join('|');
  }

  private buildHeatmapCanvas(
    masses: Float32Array,
    width: number,
    height: number
  ): HTMLCanvasElement | null {
    const { grid, n, p3, p97 } = potentialOnGrid(masses, this.config.eps, this.config.G, this.config.c, 64);
    const span = p97 - p3 || 1.0;

    const combined = document.createElement('canvas');
    combined.width = width;
    combined.height = height;
    const combinedCtx = combined.getContext('2d');
    if (!combinedCtx) return null;

    // Render the heatmap image data into a small buffer, then scale it into
    // the combined layer canvas.
    const offCanvas = document.createElement('canvas');
    offCanvas.width = n;
    offCanvas.height = n;
    const offCtx = offCanvas.getContext('2d');
    if (offCtx) {
      const imgData = offCtx.createImageData(n, n);
      const data = imgData.data;

      const HEAT_PALETTE: ReadonlyArray<readonly [number, number, number]> = [
        [18, 26, 75],
        [58, 45, 86],
        [98, 66, 88],
        [140, 88, 77],
        [185, 112, 60],
        [230, 140, 45],
      ];
      for (let gy = 0; gy < n; gy++) {
        const invGy = n - 1 - gy;
        for (let gx = 0; gx < n; gx++) {
          const val = grid[invGy * n + gx];
          const t = Math.max(0, Math.min(1, (val - p3) / span));
          const band = Math.min(5, Math.floor(t * 6));
          const [r, g, b] = HEAT_PALETTE[band];
          const alpha = 35 + band * 9;

          const idx = (gy * n + gx) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = alpha;
        }
      }
      offCtx.putImageData(imgData, 0, 0);
      combinedCtx.drawImage(offCanvas, 0, 0, width, height);
    }

    // Equipotential contours are drawn into the same cached layer so they are
    // not recomputed during pulse ticks either.
    if (this.layers.showContours) {
      const levels: number[] = [];
      const numLevels = 14;
      for (let k = 1; k < numLevels; k++) {
        levels.push(p3 + (span * k) / numLevels);
      }

      const contours = generateContourLines(grid, n, levels);
      combinedCtx.save();
      combinedCtx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      combinedCtx.lineWidth = 1.2;

      contours.forEach((segments) => {
        combinedCtx.beginPath();
        segments.forEach((seg) => {
          const x1 = ((seg.x1 + 0.5) / 8.0) * width;
          const y1 = height - ((seg.y1 + 0.5) / 8.0) * height;
          const x2 = ((seg.x2 + 0.5) / 8.0) * width;
          const y2 = height - ((seg.y2 + 0.5) / 8.0) * height;
          combinedCtx.moveTo(x1, y1);
          combinedCtx.lineTo(x2, y2);
        });
        combinedCtx.stroke();
      });
      combinedCtx.restore();
    }

    return combined;
  }

  public render(): void {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    const sqSize = width / 8.0;
    const flip = this.orientation === 'b';
    const disp = (sq: number): number => (flip ? 63 - sq : sq);
    const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 350);

    // 1. Draw Board Base Squares
    const sqLight = '#e9eff4';
    const sqDark = '#b8c9d9';

    for (let r = 0; r < 8; r++) {
      for (let f = 0; f < 8; f++) {
        const isLight = (r + f) % 2 === 0;
        ctx.fillStyle = isLight ? sqLight : sqDark;
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    // 2. Draw Hover Square Highlight
    if (this.hoveredSquare !== null && !this.isDragging) {
      const f = this.hoveredSquare % 8;
      const r = Math.floor(this.hoveredSquare / 8);
      ctx.fillStyle = 'rgba(36, 72, 184, 0.18)'; // Ultramarine highlight
      ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
    }

    // 3. Draw Selected Square (Tap-to-move) Highlight
    if (this.selectedSquare !== null) {
      const f = this.selectedSquare % 8;
      const r = Math.floor(this.selectedSquare / 8);
      ctx.save();
      ctx.fillStyle = 'rgba(240, 100, 38, 0.25)';
      ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      ctx.strokeStyle = '#f06426';
      ctx.lineWidth = 3;
      ctx.strokeRect(f * sqSize + 1.5, (7 - r) * sqSize + 1.5, sqSize - 3, sqSize - 3);
      ctx.restore();
    }

    // 4. Draw Last Move Highlight
    if (this.lastMove) {
      ctx.fillStyle = 'rgba(240, 100, 38, 0.35)'; // Trajectory Orange
      for (const sq of [disp(this.lastMove.from), disp(this.lastMove.to)]) {
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    const rawMasses = this.board.massVector();
    let masses: Float32Array = flip
      ? Float32Array.from({ length: 64 }, (_, d) => rawMasses[63 - d])
      : rawMasses;

    // Live field warping: relocate the dragged piece's mass under the cursor.
    if (this.isDragging && this.dragPiece && this.dragFromSq !== null) {
      masses = Float32Array.from(masses);
      const targetSq = this.hoveredSquare ?? this.dragFromSq;
      const carried = masses[this.dragFromSq];
      masses[this.dragFromSq] = 0;
      if (targetSq !== null) masses[targetSq] += carried;
    }

    const cacheKey = this.buildCacheKey(width, height);
    const cacheChanged = cacheKey !== this.cacheKey;

    // 4. Draw Continuous Plummer Potential Heatmap (+ Equipotential Contours)
    if (this.layers.showHeatmap) {
      if (cacheChanged || !this.heatmapCanvas) {
        this.heatmapCanvas = this.buildHeatmapCanvas(masses, width, height);
      }
      if (this.heatmapCanvas) {
        ctx.drawImage(this.heatmapCanvas, 0, 0);
      }
    } else {
      this.heatmapCanvas = null;
    }

    // 6. Draw Gravitational Force Vectors (Streamlines)
    if (this.layers.showVectors) {
      let forces: { fx: Float32Array; fy: Float32Array };
      if (cacheChanged || !this.cachedForces) {
        forces = forceField(masses, this.config.eps, this.config.G, this.config.c);
        this.cachedForces = forces;
      } else {
        forces = this.cachedForces;
      }

      ctx.save();
      ctx.strokeStyle = 'rgba(240, 100, 38, 0.65)';
      ctx.fillStyle = 'rgba(240, 100, 38, 0.65)';
      ctx.lineWidth = 1.4;

      for (let sq = 0; sq < 64; sq++) {
        const fx = forces.fx[sq];
        const fy = forces.fy[sq];
        const mag = Math.sqrt(fx * fx + fy * fy);
        if (mag < 0.05) continue;

        const f = sq % 8;
        const r = Math.floor(sq / 8);
        const cx = (f + 0.5) * sqSize;
        const cy = (7 - r + 0.5) * sqSize;

        const normX = fx / (mag + 1e-6);
        const normY = fy / (mag + 1e-6);
        const len = Math.min(sqSize * 0.4, sqSize * 0.15 * Math.log10(1 + mag * 5));

        const endX = cx + normX * len;
        const endY = cy - normY * len; // Invert Y for canvas

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Arrow head
        const angle = Math.atan2(-normY, normX);
        const head = 4;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - head * Math.cos(angle - Math.PI / 6), endY - head * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(endX - head * Math.cos(angle + Math.PI / 6), endY - head * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

    // 7. Evaluate Position and Render Tidal Tensors
    let breakdown: ScoreBreakdown;
    let breakdownFreshlyComputed = false;
    if (cacheChanged || !this.cachedBreakdown) {
      breakdown = evaluatePosition(this.board, this.config);
      this.cachedBreakdown = breakdown;
      breakdownFreshlyComputed = true;
    } else {
      breakdown = this.cachedBreakdown;
    }

    this.cacheKey = cacheKey;

    if (breakdownFreshlyComputed && this.onEvaluateCallback) {
      this.onEvaluateCallback(breakdown);
    }

    if (this.layers.showTidalStress) {
      const wk = this.board.findKingSquare('w');
      const bk = this.board.findKingSquare('b');

      if (wk !== null && breakdown.whiteKingTidal) {
        const dk = disp(wk);
        const kx = (dk % 8 + 0.5) * sqSize;
        const ky = (7 - Math.floor(dk / 8) + 0.5) * sqSize;
        drawTidalStressEllipse(ctx, kx, ky, sqSize, breakdown.whiteKingTidal);
      }

      if (bk !== null && breakdown.blackKingTidal) {
        const dk = disp(bk);
        const kx = (dk % 8 + 0.5) * sqSize;
        const ky = (7 - Math.floor(dk / 8) + 0.5) * sqSize;
        drawTidalStressEllipse(ctx, kx, ky, sqSize, breakdown.blackKingTidal);
      }
    }

    // 7b. Draw Retarded Gravitational Wavefronts (finite c)
    if (this.layers.showWavefronts && this.lastMove) {
      const origin = disp(this.lastMove.from);
      const of = origin % 8;
      const or = Math.floor(origin / 8);
      const cx = (of + 0.5) * sqSize;
      const cy = ((7 - or) + 0.5) * sqSize;
      const cSq = this.config.c;

      ctx.save();
      const ripples = [1, 0.62, 0.3];
      ripples.forEach((f, idx) => {
        ctx.beginPath();
        ctx.arc(cx, cy, cSq * f * sqSize, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 105, 255, ${((0.5 - idx * 0.14) * (0.7 + 0.3 * pulse)).toFixed(3)})`;
        ctx.lineWidth = idx === 0 ? 1.6 : 1.2;
        ctx.setLineDash(idx === 0 ? [5, 5] : []);
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // Horizon indicator: squares outside the light cone have not felt the update.
      ctx.strokeStyle = 'rgba(17, 17, 17, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (let sq = 0; sq < 64; sq++) {
        if (DIST_64[origin * 64 + sq] <= cSq) continue;
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.strokeRect(f * sqSize + 1.5, (7 - r) * sqSize + 1.5, sqSize - 3, sqSize - 3);
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    // 7c. Draw Kinetic Trails (Lorentz escalation)
    if (this.layers.showLorentz && this.lastMove) {
      const from = disp(this.lastMove.from);
      const to = disp(this.lastMove.to);
      const dist = DIST_64[from * 64 + to];
      if (dist >= 2) {
        const ratio = Math.min(0.95, dist / this.config.c);
        const rel = ratio > 0.6;
        const ff = from % 8;
        const fr = Math.floor(from / 8);
        const tf = to % 8;
        const tr = Math.floor(to / 8);
        const sx = (ff + 0.5) * sqSize;
        const sy = ((7 - fr) + 0.5) * sqSize;
        const ex = (tf + 0.5) * sqSize;
        const ey = ((7 - tr) + 0.5) * sqSize;

        ctx.save();
        const strokes = rel ? 4 : 2;
        for (let i = 1; i <= strokes; i++) {
          const t = i / (strokes + 1);
          ctx.beginPath();
          ctx.moveTo(sx, sy);
          ctx.lineTo(sx + (ex - sx) * t, sy + (ey - sy) * t);
          ctx.strokeStyle = `rgba(0, 229, 255, ${(rel ? 0.4 : 0.18) * (1 - t)})`;
          ctx.lineWidth = rel ? 2.2 : 1.2;
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // 7d. Draw Lagrange Equilibrium Points
    if (this.layers.showLagrange) {
      const lp = this.computeLagrangePoints();
      ctx.save();
      lp.forEach((p) => {
        const px = (flip ? 8 - p.x : p.x) * sqSize;
        const py = (flip ? p.y : 8 - p.y) * sqSize;
        const triangular = p.label === 'L4' || p.label === 'L5';
        const arm = sqSize * 0.18;

        ctx.beginPath();
        ctx.moveTo(px - arm, py);
        ctx.lineTo(px + arm, py);
        ctx.moveTo(px, py - arm);
        ctx.lineTo(px, py + arm);
        ctx.strokeStyle = triangular ? 'rgba(255, 90, 0, 0.6)' : 'rgba(0, 105, 255, 0.65)';
        ctx.lineWidth = 1.4;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fillStyle = triangular ? 'rgba(255, 90, 0, 0.8)' : 'rgba(0, 105, 255, 0.8)';
        ctx.fill();

        ctx.font = '700 9px Sometype Mono, monospace';
        ctx.fillStyle = 'rgba(17, 17, 17, 0.6)';
        ctx.fillText(p.label, px + 3, py - 3);
      });
      ctx.restore();
    }

    // 7e. Draw Accretion Halos (Layer 2)
    if (this.layers.showAccretion) {
      ctx.save();
      ctx.globalAlpha = 0.7 + 0.3 * pulse;
      drawAccretionHalos(ctx, this.board, this.accretionExcess, sqSize, flip);
      ctx.restore();
    }

    // 8. Draw Pieces
    for (let sq = 0; sq < 64; sq++) {
      if (this.isDragging && sq === this.dragFromSq) continue;
      const piece = this.board.squares[disp(sq)];
      if (!piece) continue;

      const f = sq % 8;
      const r = Math.floor(sq / 8);
      const px = f * sqSize;
      const py = (7 - r) * sqSize;

      const imgKey = `${piece.color}${piece.type}`;
      const img = this.pieceImages.get(imgKey);
      if (img && this.isLoaded) {
        const piecePad = sqSize * 0.08;
        ctx.drawImage(img, px + piecePad, py + piecePad, sqSize - 2 * piecePad, sqSize - 2 * piecePad);
      }

      // Draw Accretion Mass Badge (Layer 2)
      if (this.layers.showAccretion && (piece.mass > (piece.type === 'k' ? 1001.0 : piece.type === 'q' ? 9.2 : 5.2))) {
        ctx.save();
        ctx.font = '600 10px Sometype Mono, monospace';
        ctx.fillStyle = '#f06426';
        ctx.textAlign = 'right';
        ctx.fillText(`${piece.mass.toFixed(1)}m`, px + sqSize - 4, py + sqSize - 4);
        ctx.restore();
      }
    }

    // 9. Draw Dragging Piece
    if (this.isDragging && this.dragPiece) {
      const imgKey = `${this.dragPiece.color}${this.dragPiece.type}`;
      const img = this.pieceImages.get(imgKey);
      if (img && this.isLoaded) {
        const drawSize = sqSize * 1.15;
        ctx.save();
        ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
        ctx.shadowBlur = 12;
        ctx.shadowOffsetY = 6;
        ctx.drawImage(
          img,
          this.dragCanvasX - drawSize / 2,
          this.dragCanvasY - drawSize / 2,
          drawSize,
          drawSize
        );
        ctx.restore();
      }
    }

    // 9b. Draw Capture Matter Stream (Layer 2)
    if (this.captureStream) {
      const t = Math.min(1, (performance.now() - this.captureStream.startTime) / 300);
      drawCaptureStream(ctx, disp(this.captureStream.fromSq), disp(this.captureStream.toSq), sqSize, t);
    }

    // 9c. Draw Relativistic Glow (Lorentz escalation)
    if (this.layers.showLorentz && this.lastMove) {
      const dist = DIST_64[this.lastMove.from * 64 + this.lastMove.to];
      const ratio = Math.min(0.95, dist / this.config.c);
      if (ratio > 0.6) {
        const to = disp(this.lastMove.to);
        const f = to % 8;
        const r = Math.floor(to / 8);
        const cx = (f + 0.5) * sqSize;
        const cy = ((7 - r) + 0.5) * sqSize;
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, sqSize * 0.58, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 229, 255, ${(0.45 + 0.35 * pulse).toFixed(3)})`;
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.restore();
      }
    }

    // 9d. Draw Gravitational Anchors (piece on a Lagrange point)
    if (this.layers.showLagrange) {
      const lp = this.computeLagrangePoints();
      for (let sq = 0; sq < 64; sq++) {
        const piece = this.board.squares[sq];
        if (!piece || (piece.type !== 'n' && piece.type !== 'b')) continue;
        const px = (sq % 8) + 0.5;
        const py = Math.floor(sq / 8) + 0.5;
        if (!lp.some((p) => Math.hypot(p.x - px, p.y - py) < 0.75)) continue;

        const dsq = disp(sq);
        const f = dsq % 8;
        const r = Math.floor(dsq / 8);
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 105, 255, 0.9)';
        ctx.lineWidth = 2;
        ctx.strokeRect(f * sqSize + 2, (7 - r) * sqSize + 2, sqSize - 4, sqSize - 4);
        ctx.font = '700 8px Sometype Mono, monospace';
        ctx.fillStyle = 'rgba(0, 105, 255, 0.95)';
        ctx.fillText('GRAV. ANCHOR', f * sqSize + 3, (7 - r) * sqSize + 11);
        ctx.restore();
      }
    }

    // 10. Board Outer Coordinates (a-h, 1-8)
    ctx.save();
    ctx.font = '600 10px Sometype Mono, monospace';
    ctx.fillStyle = 'rgba(23, 34, 53, 0.55)';

    for (let f = 0; f < 8; f++) {
      const char = String.fromCharCode(97 + (flip ? 7 - f : f));
      ctx.fillText(char, f * sqSize + 3, height - 3);
    }
    for (let r = 0; r < 8; r++) {
      const num = (flip ? 8 - r : r + 1).toString();
      ctx.fillText(num, width - 10, (7 - r) * sqSize + 12);
    }
    ctx.restore();
  }
}
