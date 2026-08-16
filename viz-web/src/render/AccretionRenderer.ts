/**
 * Kepler-64 Accretion Renderer
 * Renders accretion halos and capture streams on the unified canvas.
 */

import type { KeplerBoard } from '../core/board';
import { haloRadius } from '../core/accretion';

export function drawAccretionHalos(
  ctx: CanvasRenderingContext2D,
  board: KeplerBoard,
  excessBySquare: Record<number, number>,
  sqSize: number,
  flip = false
): void {
  for (let sq = 0; sq < 64; sq++) {
    const excess = excessBySquare[sq];
    if (!excess || excess <= 0) continue;
    if (!board.squares[sq]) continue;

    const dsq = flip ? 63 - sq : sq;
    const f = dsq % 8;
    const r = Math.floor(dsq / 8);
    const cx = (f + 0.5) * sqSize;
    const cy = ((7 - r) + 0.5) * sqSize;
    const radius = haloRadius(excess) * sqSize;

    // Inner ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0, 105, 255, 0.55)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Outer ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 90, 0, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

export function drawCaptureStream(
  ctx: CanvasRenderingContext2D,
  fromSq: number,
  toSq: number,
  sqSize: number,
  t: number
): void {
  const fromF = fromSq % 8;
  const fromR = Math.floor(fromSq / 8);
  const toF = toSq % 8;
  const toR = Math.floor(toSq / 8);

  const startX = (fromF + 0.5) * sqSize;
  const startY = ((7 - fromR) + 0.5) * sqSize;
  const endX = (toF + 0.5) * sqSize;
  const endY = ((7 - toR) + 0.5) * sqSize;

  // Control point offset perpendicular to the straight line.
  const dx = endX - startX;
  const dy = endY - startY;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const ctrlX = (startX + endX) / 2 + nx * sqSize * 0.35;
  const ctrlY = (startY + endY) / 2 + ny * sqSize * 0.35;

  const particleRadius = sqSize * 0.06;

  for (let i = 0; i < 5; i++) {
    const s = i / 4;
    if (t < s) continue;

    // Quadratic bezier from start to end through the control point.
    const inv = 1 - s;
    const x = inv * inv * startX + 2 * inv * s * ctrlX + s * s * endX;
    const y = inv * inv * startY + 2 * inv * s * ctrlY + s * s * endY;

    const alpha = 0.85 - (0.85 - 0.1) * s;

    ctx.beginPath();
    ctx.arc(x, y, particleRadius, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 90, 0, ${alpha.toFixed(3)})`;
    ctx.fill();
  }
}
