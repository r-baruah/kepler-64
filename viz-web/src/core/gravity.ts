/**
 * Kepler-64 N-body Plummer Gravitational Field Calculator
 */

export interface Point2D {
  x: number;
  y: number;
}

// 64 discrete coordinates on the 8x8 chessboard
export const COORDS_64: Point2D[] = [];
for (let i = 0; i < 64; i++) {
  COORDS_64.push({
    x: i % 8,
    y: Math.floor(i / 8),
  });
}

// Precomputed 64x64 distance matrices
export const DIST2_64 = new Float32Array(64 * 64);
export const DIST_64 = new Float32Array(64 * 64);
export const DIFF_X_64 = new Float32Array(64 * 64);
export const DIFF_Y_64 = new Float32Array(64 * 64);

for (let i = 0; i < 64; i++) {
  const ci = COORDS_64[i];
  for (let j = 0; j < 64; j++) {
    const cj = COORDS_64[j];
    const dx = ci.x - cj.x;
    const dy = ci.y - cj.y;
    const idx = i * 64 + j;
    DIFF_X_64[idx] = dx;
    DIFF_Y_64[idx] = dy;
    const d2 = dx * dx + dy * dy;
    DIST2_64[idx] = d2;
    DIST_64[idx] = Math.sqrt(d2);
  }
}

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

/**
 * Computes scalar Plummer gravitational potential at all 64 board squares.
 * U_i = -G * sum_j [ |m_j| * sigmoid(c - d_ij) / sqrt(d_ij^2 + eps^2) ]
 */
export function potentialField(
  masses: Float32Array,
  eps: number,
  G: number,
  c: number
): Float32Array {
  const U = new Float32Array(64);
  const eps2 = eps * eps;

  for (let i = 0; i < 64; i++) {
    let sum = 0;
    const iOffset = i * 64;
    for (let j = 0; j < 64; j++) {
      const absM = Math.abs(masses[j]);
      if (absM < 1e-6) continue;
      const d = DIST_64[iOffset + j];
      const gate = sigmoid(c - d);
      const r = Math.sqrt(DIST2_64[iOffset + j] + eps2);
      sum += (absM * gate) / r;
    }
    U[i] = -G * sum;
  }
  return U;
}

/**
 * Computes 2D gravitational force vector at all 64 board squares.
 * F_i = G * sum_j [ |m_j| * sigmoid(c - d_ij) * (r_i - r_j) / (d_ij^2 + eps^2)^(3/2) ]
 */
export function forceField(
  masses: Float32Array,
  eps: number,
  G: number,
  c: number
): { fx: Float32Array; fy: Float32Array } {
  const fx = new Float32Array(64);
  const fy = new Float32Array(64);
  const eps2 = eps * eps;

  for (let i = 0; i < 64; i++) {
    let sumX = 0;
    let sumY = 0;
    const iOffset = i * 64;
    for (let j = 0; j < 64; j++) {
      const absM = Math.abs(masses[j]);
      if (absM < 1e-6) continue;
      const d = DIST_64[iOffset + j];
      const gate = sigmoid(c - d);
      const r2 = DIST2_64[iOffset + j] + eps2;
      const invR3 = gate / (r2 * Math.sqrt(r2));
      const idx = iOffset + j;
      sumX += absM * DIFF_X_64[idx] * invR3;
      sumY += absM * DIFF_Y_64[idx] * invR3;
    }
    fx[i] = G * sumX;
    fy[i] = G * sumY;
  }
  return { fx, fy };
}

/**
 * Evaluates continuous Plummer potential on an n x n spatial grid.
 * Used for high-resolution heatmap and contour visualization.
 */
export function potentialOnGrid(
  masses: Float32Array,
  eps: number,
  G: number,
  c: number,
  n: number = 72
): { grid: Float32Array; n: number; min: number; max: number; p3: number; p97: number } {
  const grid = new Float32Array(n * n);
  const eps2 = eps * eps;
  const dx = 8.0 / n;
  const halfDx = dx / 2.0;

  // Track active pieces for fast evaluation
  const activeMasses: { x: number; y: number; m: number }[] = [];
  for (let j = 0; j < 64; j++) {
    const absM = Math.abs(masses[j]);
    if (absM > 1e-6) {
      activeMasses.push({
        x: j % 8,
        y: Math.floor(j / 8),
        m: absM,
      });
    }
  }

  let min = Infinity;
  let max = -Infinity;

  for (let gy = 0; gy < n; gy++) {
    const y = -0.5 + halfDx + gy * dx;
    const yOffset = gy * n;
    for (let gx = 0; gx < n; gx++) {
      const x = -0.5 + halfDx + gx * dx;
      let sum = 0;
      for (let k = 0; k < activeMasses.length; k++) {
        const p = activeMasses[k];
        const dist2 = (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
        const dist = Math.sqrt(dist2);
        const gate = sigmoid(c - dist);
        const r = Math.sqrt(dist2 + eps2);
        sum += (p.m * gate) / r;
      }
      const val = -G * sum;
      grid[yOffset + gx] = val;
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }

  // Calculate 3rd and 97th percentiles for visual balance
  const sorted = Float32Array.from(grid).sort();
  const p3 = sorted[Math.floor(sorted.length * 0.03)] ?? min;
  const p97 = sorted[Math.floor(sorted.length * 0.97)] ?? max;

  return { grid, n, min, max, p3, p97 };
}
