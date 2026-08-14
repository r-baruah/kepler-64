/**
 * Kepler-64 Unified Canvas Renderer
 * Renders the fused Chessboard + Topographic Gravity Field + Force Vectors + Tidal Tensors
 */

import { KeplerBoard } from '../core/board';
import type { Piece } from '../core/board';
import type { ConstantsConfig } from '../core/constants';
import { DEFAULT_CONSTANTS } from '../core/constants';
import { potentialOnGrid, forceField } from '../core/gravity';
import { evaluatePosition } from '../core/evaluate';
import type { ScoreBreakdown } from '../core/evaluate';
import { generateContourLines } from './ContourRenderer';
import { drawTidalStressEllipse } from './TidalRenderer';

export interface RenderLayers {
  showHeatmap: boolean;
  showContours: boolean;
  showVectors: boolean;
  showTidalStress: boolean;
  showAccretion: boolean;
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
  };

  private pieceImages: Map<string, HTMLImageElement> = new Map();
  private isLoaded = false;
  private lastMove: { from: number; to: number } | null = null;

  // Hover state
  private hoveredSquare: number | null = null;
  private onHoverCallback?: (info: SquareHoverInfo | null) => void;

  // Drag-and-drop state
  private isDragging = false;
  private dragPiece: Piece | null = null;
  private dragFromSq: number | null = null;
  private dragCurrentX = 0;
  private dragCurrentY = 0;
  private onMoveCallback?: (fromSq: number, toSq: number) => void;
  private onEvaluateCallback?: (breakdown: ScoreBreakdown) => void;

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
  }

  public setBoard(board: KeplerBoard, lastMove: { from: number; to: number } | null = null): void {
    this.board = board;
    this.lastMove = lastMove;
    this.render();
  }

  public setConfig(config: ConstantsConfig): void {
    this.config = config;
    this.render();
  }

  public setLayers(layers: Partial<RenderLayers>): void {
    this.layers = { ...this.layers, ...layers };
    this.render();
  }

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
    this.canvas.addEventListener('mousedown', this.handlePointerDown.bind(this));
    this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
    this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
    window.addEventListener('mousemove', this.handleWindowMouseMove.bind(this));
    window.addEventListener('mouseup', this.handlePointerUp.bind(this));

    // Touch support
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        const touch = e.touches[0];
        this.startDrag(touch.clientX, touch.clientY);
      }
    }, { passive: false });

    window.addEventListener('touchmove', (e) => {
      if (this.isDragging && e.touches.length === 1) {
        e.preventDefault();
        const touch = e.touches[0];
        this.updateDrag(touch.clientX, touch.clientY);
      }
    }, { passive: false });

    window.addEventListener('touchend', () => {
      if (this.isDragging) {
        this.endDrag();
      }
    });
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
          const f = sq % 8;
          const r = Math.floor(sq / 8);
          const piece = this.board.squares[sq];
          const masses = this.board.massVector();
          const forces = forceField(masses, this.config.eps, this.config.G, this.config.c);
          const fMag = Math.sqrt(forces.fx[sq] * forces.fx[sq] + forces.fy[sq] * forces.fy[sq]);

          this.onHoverCallback({
            square: sq,
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
    const sq = this.getCanvasSquare(clientX, clientY);
    if (sq === null) return;
    const piece = this.board.squares[sq];
    if (!piece) return;

    this.isDragging = true;
    this.dragPiece = piece;
    this.dragFromSq = sq;
    const rect = this.canvas.getBoundingClientRect();
    this.dragCurrentX = clientX - rect.left;
    this.dragCurrentY = clientY - rect.top;
    this.render();
  }

  private handleWindowMouseMove(e: MouseEvent): void {
    if (!this.isDragging) return;
    this.updateDrag(e.clientX, e.clientY);
  }

  private updateDrag(clientX: number, clientY: number): void {
    const rect = this.canvas.getBoundingClientRect();
    this.dragCurrentX = clientX - rect.left;
    this.dragCurrentY = clientY - rect.top;
    this.render();
  }

  private handlePointerUp(_e: MouseEvent): void {
    if (!this.isDragging) return;
    this.endDrag();
  }

  private endDrag(): void {
    if (!this.isDragging) return;
    const targetSq = this.getCanvasSquare(
      this.canvas.getBoundingClientRect().left + this.dragCurrentX,
      this.canvas.getBoundingClientRect().top + this.dragCurrentY
    );

    if (targetSq !== null && this.dragFromSq !== null && targetSq !== this.dragFromSq) {
      if (this.onMoveCallback) {
        this.onMoveCallback(this.dragFromSq, targetSq);
      } else {
        // Direct local board move
        this.board.squares[targetSq] = this.dragPiece;
        this.board.squares[this.dragFromSq] = null;
        this.lastMove = { from: this.dragFromSq, to: targetSq };
      }
    }

    this.isDragging = false;
    this.dragPiece = null;
    this.dragFromSq = null;
    this.render();
  }

  public render(): void {
    const width = this.canvas.width;
    const height = this.canvas.height;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    const sqSize = width / 8.0;

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

    // 3. Draw Last Move Highlight
    if (this.lastMove) {
      ctx.fillStyle = 'rgba(240, 100, 38, 0.35)'; // Trajectory Orange
      for (const sq of [this.lastMove.from, this.lastMove.to]) {
        const f = sq % 8;
        const r = Math.floor(sq / 8);
        ctx.fillRect(f * sqSize, (7 - r) * sqSize, sqSize, sqSize);
      }
    }

    const masses = this.board.massVector();

    // 4. Draw Continuous Plummer Potential Heatmap
    if (this.layers.showHeatmap) {
      const { grid, n, p3, p97 } = potentialOnGrid(masses, this.config.eps, this.config.G, this.config.c, 64);
      const span = p97 - p3 || 1.0;

      const offCanvas = document.createElement('canvas');
      offCanvas.width = n;
      offCanvas.height = n;
      const offCtx = offCanvas.getContext('2d');
      if (offCtx) {
        const imgData = offCtx.createImageData(n, n);
        const data = imgData.data;

        for (let gy = 0; gy < n; gy++) {
          const invGy = n - 1 - gy;
          for (let gx = 0; gx < n; gx++) {
            const val = grid[invGy * n + gx];
            const t = Math.max(0, Math.min(1, (val - p3) / span));

            // Deep Ultramarine -> Indigo -> Soft Amber -> Light Magma
            const r = Math.round(18 + t * (230 - 18));
            const g = Math.round(26 + t * (140 - 26));
            const b = Math.round(75 + t * (45 - 75));
            const alpha = Math.round(35 + t * 45); // Subtle, non-obscuring 15-30% opacity

            const idx = (gy * n + gx) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = alpha;
          }
        }
        offCtx.putImageData(imgData, 0, 0);

        ctx.save();
        ctx.drawImage(offCanvas, 0, 0, width, height);
        ctx.restore();
      }

      // 5. Draw Equipotential Contour Lines
      if (this.layers.showContours) {
        const levels: number[] = [];
        const numLevels = 14;
        for (let k = 1; k < numLevels; k++) {
          levels.push(p3 + (span * k) / numLevels);
        }

        const contours = generateContourLines(grid, n, levels);
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.lineWidth = 1.2;

        contours.forEach((segments) => {
          ctx.beginPath();
          segments.forEach((seg) => {
            const x1 = ((seg.x1 + 0.5) / 8.0) * width;
            const y1 = height - ((seg.y1 + 0.5) / 8.0) * height;
            const x2 = ((seg.x2 + 0.5) / 8.0) * width;
            const y2 = height - ((seg.y2 + 0.5) / 8.0) * height;
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
          });
          ctx.stroke();
        });
        ctx.restore();
      }
    }

    // 6. Draw Gravitational Force Vectors (Streamlines)
    if (this.layers.showVectors) {
      const forces = forceField(masses, this.config.eps, this.config.G, this.config.c);
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
    const breakdown = evaluatePosition(this.board, this.config);
    if (this.onEvaluateCallback) {
      this.onEvaluateCallback(breakdown);
    }

    if (this.layers.showTidalStress) {
      const wk = this.board.findKingSquare('w');
      const bk = this.board.findKingSquare('b');

      if (wk !== null && breakdown.whiteKingTidal) {
        const kx = (wk % 8 + 0.5) * sqSize;
        const ky = (7 - Math.floor(wk / 8) + 0.5) * sqSize;
        drawTidalStressEllipse(ctx, kx, ky, sqSize, breakdown.whiteKingTidal);
      }

      if (bk !== null && breakdown.blackKingTidal) {
        const kx = (bk % 8 + 0.5) * sqSize;
        const ky = (7 - Math.floor(bk / 8) + 0.5) * sqSize;
        drawTidalStressEllipse(ctx, kx, ky, sqSize, breakdown.blackKingTidal);
      }
    }

    // 8. Draw Pieces
    for (let sq = 0; sq < 64; sq++) {
      if (this.isDragging && sq === this.dragFromSq) continue;
      const piece = this.board.squares[sq];
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
          this.dragCurrentX - drawSize / 2,
          this.dragCurrentY - drawSize / 2,
          drawSize,
          drawSize
        );
        ctx.restore();
      }
    }

    // 10. Board Outer Coordinates (a-h, 1-8)
    ctx.save();
    ctx.font = '600 10px Sometype Mono, monospace';
    ctx.fillStyle = 'rgba(23, 34, 53, 0.55)';

    for (let f = 0; f < 8; f++) {
      const char = String.fromCharCode(97 + f);
      ctx.fillText(char, f * sqSize + 3, height - 3);
    }
    for (let r = 0; r < 8; r++) {
      const num = (r + 1).toString();
      ctx.fillText(num, width - 10, (7 - r) * sqSize + 12);
    }
    ctx.restore();
  }
}
