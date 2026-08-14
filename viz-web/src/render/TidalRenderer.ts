/**
 * Kepler-64 King Tidal Stress Ellipse & Disruption Vector Renderer
 */

import type { TidalDisruptionResult } from '../core/tidal';

export function drawTidalStressEllipse(
  ctx: CanvasRenderingContext2D,
  kingX: number, // pixel center X
  kingY: number, // pixel center Y
  sqSize: number,
  tidal: TidalDisruptionResult,
  maxLam: number = 3.0
): void {
  const normLam1 = Math.min(2.5, Math.abs(tidal.lambda1) / (maxLam + 1e-6));
  const normLam2 = Math.min(2.5, Math.abs(tidal.lambda2) / (maxLam + 1e-6));

  const scale = sqSize * 0.75;
  const radiusX = Math.max(scale * normLam1, sqSize * 0.22);
  const radiusY = Math.max(scale * normLam2, sqSize * 0.12);
  const angleRad = (tidal.angleDeg * Math.PI) / 180.0;

  ctx.save();
  ctx.translate(kingX, kingY);
  ctx.rotate(angleRad);

  // 1. Fill ellipse with soft gradient
  ctx.beginPath();
  ctx.ellipse(0, 0, radiusX, radiusY, 0, 0, Math.PI * 2);
  ctx.fillStyle = tidal.isDisrupted ? 'rgba(220, 38, 38, 0.22)' : 'rgba(43, 144, 204, 0.18)';
  ctx.fill();

  // 2. Stroke boundary
  ctx.lineWidth = tidal.isDisrupted ? 2.5 : 1.5;
  ctx.strokeStyle = tidal.color;
  ctx.stroke();

  // 3. Principal line-of-failure vector arrow
  if (tidal.isDisrupted || Math.abs(tidal.lambda1) > 0.4) {
    const arrowLen = radiusX * 1.35;
    ctx.beginPath();
    ctx.moveTo(-arrowLen * 0.6, 0);
    ctx.lineTo(arrowLen * 0.6, 0);
    ctx.strokeStyle = tidal.isDisrupted ? '#dc2626' : tidal.color;
    ctx.lineWidth = 2.0;
    ctx.stroke();

    // Arrow heads on both ends of stretching axis
    const headSize = 5;
    ctx.fillStyle = tidal.isDisrupted ? '#dc2626' : tidal.color;
    
    // Positive end
    ctx.beginPath();
    ctx.moveTo(arrowLen * 0.6, 0);
    ctx.lineTo(arrowLen * 0.6 - headSize, -headSize * 0.7);
    ctx.lineTo(arrowLen * 0.6 - headSize, headSize * 0.7);
    ctx.closePath();
    ctx.fill();

    // Negative end
    ctx.beginPath();
    ctx.moveTo(-arrowLen * 0.6, 0);
    ctx.lineTo(-arrowLen * 0.6 + headSize, -headSize * 0.7);
    ctx.lineTo(-arrowLen * 0.6 + headSize, headSize * 0.7);
    ctx.closePath();
    ctx.fill();
  }

  ctx.restore();
}
