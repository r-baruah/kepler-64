/**
 * Kepler-64 Tidal Tensor, Closed-Form Eigensystem & Roche Disruption
 */

export interface EigensystemResult {
  lambda1: number;
  lambda2: number;
  angleRad: number;
  angleDeg: number;
  v1: { x: number; y: number };
}

export interface TidalDisruptionResult {
  eta: number;
  lambda1: number;
  lambda2: number;
  angleDeg: number;
  v1: { x: number; y: number };
  tensor: { uFile: number; uRank: number; uxy: number };
  isDisrupted: boolean;
  color: string;
}

/**
 * Closed-form analytical eigenvalues and principal eigenvector of 2x2 symmetric matrix.
 * Matrix A = [ [uFile, uxy], [uxy, uRank] ]
 */
export function eigensystem2x2(uFile: number, uRank: number, uxy: number): EigensystemResult {
  const tr = uFile + uRank;
  const det = uFile * uRank - uxy * uxy;
  const disc = Math.max(tr * tr / 4.0 - det, 1e-12);
  const s = Math.sqrt(disc);

  const lambda1 = tr / 2.0 + s;
  const lambda2 = tr / 2.0 - s;

  // Principal eigenvector for lambda1: (A - lambda1 * I) * v = 0
  let vx = 1.0;
  let vy = 0.0;

  if (Math.abs(uxy) > 1e-7) {
    vx = lambda1 - uRank;
    vy = uxy;
  } else if (Math.abs(uFile - uRank) > 1e-7) {
    if (uFile > uRank) {
      vx = 1.0;
      vy = 0.0;
    } else {
      vx = 0.0;
      vy = 1.0;
    }
  }

  const norm = Math.sqrt(vx * vx + vy * vy) || 1.0;
  vx /= norm;
  vy /= norm;

  const angleRad = Math.atan2(vy, vx);
  const angleDeg = (angleRad * 180.0) / Math.PI;

  return {
    lambda1,
    lambda2,
    angleRad,
    angleDeg,
    v1: { x: vx, y: vy },
  };
}

/**
 * Computes 2D finite-difference Hessian tensor at king square with 10x10 edge padding.
 */
export function tidalTensorAt(
  U64: Float32Array,
  kingSq: number
): { uFile: number; uRank: number; uxy: number } {
  const r = Math.floor(kingSq / 8);
  const f = kingSq % 8;

  // Pad to 10x10 with edge mode
  const padded = new Float32Array(10 * 10);
  for (let pr = 0; pr < 10; pr++) {
    const origR = Math.max(0, Math.min(7, pr - 1));
    const prOffset = pr * 10;
    for (let pf = 0; pf < 10; pf++) {
      const origF = Math.max(0, Math.min(7, pf - 1));
      padded[prOffset + pf] = U64[origR * 8 + origF];
    }
  }

  const R = r + 1;
  const F = f + 1;

  const uFile = padded[R * 10 + (F + 1)] - 2 * padded[R * 10 + F] + padded[R * 10 + (F - 1)];
  const uRank = padded[(R + 1) * 10 + F] - 2 * padded[R * 10 + F] + padded[(R - 1) * 10 + F];
  const uxy = (
    padded[(R + 1) * 10 + (F + 1)] -
    padded[(R + 1) * 10 + (F - 1)] -
    padded[(R - 1) * 10 + (F + 1)] +
    padded[(R - 1) * 10 + (F - 1)]
  ) / 4.0;

  return { uFile, uRank, uxy };
}

/**
 * Maps ratio eta / roche to RGB color: Steel Blue (safe) -> Amber -> Crimson (disrupted).
 */
export function disruptionColor(eta: number, roche: number): string {
  const ratio = Math.max(0, Math.min(1.5, eta / (roche + 1e-9)));
  if (ratio < 0.5) {
    const s = ratio * 2.0; // 0 to 1
    // Steel blue (43, 144, 204) -> Amber (235, 160, 20)
    const r = Math.round(43 + s * (235 - 43));
    const g = Math.round(144 + s * (160 - 144));
    const b = Math.round(204 + s * (20 - 204));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const s = Math.min(1.0, (ratio - 0.5) * 2.0); // 0 to 1
    // Amber (235, 160, 20) -> Crimson (220, 38, 38)
    const r = Math.round(235 + s * (220 - 235));
    const g = Math.round(160 + s * (38 - 160));
    const b = Math.round(20 + s * (38 - 20));
    return `rgb(${r}, ${g}, ${b})`;
  }
}

/**
 * Calculates tidal disruption index eta and eigensystem for a given king square.
 */
export function computeTidalDisruption(
  U64: Float32Array,
  kingSq: number,
  roche: number = 0.8,
  Rg: number = 1.0,
  mref: number = 3.5
): TidalDisruptionResult {
  const tensor = tidalTensorAt(U64, kingSq);
  const eigen = eigensystem2x2(tensor.uFile, tensor.uRank, tensor.uxy);
  const eta = (Math.pow(Rg, 3) * eigen.lambda1) / (mref * mref + 1e-9);
  const isDisrupted = eta >= roche;
  const color = disruptionColor(eta, roche);

  return {
    eta,
    lambda1: eigen.lambda1,
    lambda2: eigen.lambda2,
    angleDeg: eigen.angleDeg,
    v1: eigen.v1,
    tensor,
    isDisrupted,
    color,
  };
}
