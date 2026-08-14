/**
 * Kepler-64 Topographic Equipotential Contour Line Generator
 * Uses 2D Marching Squares to extract smooth isolines from Plummer potential grid.
 */

export interface ContourSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export function generateContourLines(
  grid: Float32Array,
  n: number,
  levels: number[]
): Map<number, ContourSegment[]> {
  const result = new Map<number, ContourSegment[]>();
  const dx = 8.0 / n;
  const halfDx = dx / 2.0;

  for (let l = 0; l < levels.length; l++) {
    const level = levels[l];
    const segments: ContourSegment[] = [];

    for (let gy = 0; gy < n - 1; gy++) {
      const y0 = -0.5 + halfDx + gy * dx;
      const y1 = y0 + dx;
      const row0 = gy * n;
      const row1 = (gy + 1) * n;

      for (let gx = 0; gx < n - 1; gx++) {
        const x0 = -0.5 + halfDx + gx * dx;
        const x1 = x0 + dx;

        const v00 = grid[row0 + gx];       // bottom-left
        const v10 = grid[row0 + gx + 1];   // bottom-right
        const v11 = grid[row1 + gx + 1];   // top-right
        const v01 = grid[row1 + gx];       // top-left

        let cellIndex = 0;
        if (v00 >= level) cellIndex |= 1;
        if (v10 >= level) cellIndex |= 2;
        if (v11 >= level) cellIndex |= 4;
        if (v01 >= level) cellIndex |= 8;

        if (cellIndex === 0 || cellIndex === 15) continue;

        // Linear interpolation helper
        const lerpBottom = () => {
          const t = (level - v00) / (v10 - v00 + 1e-12);
          return { x: x0 + t * (x1 - x0), y: y0 };
        };
        const lerpRight = () => {
          const t = (level - v10) / (v11 - v10 + 1e-12);
          return { x: x1, y: y0 + t * (y1 - y0) };
        };
        const lerpTop = () => {
          const t = (level - v01) / (v11 - v01 + 1e-12);
          return { x: x0 + t * (x1 - x0), y: y1 };
        };
        const lerpLeft = () => {
          const t = (level - v00) / (v01 - v00 + 1e-12);
          return { x: x0, y: y0 + t * (y1 - y0) };
        };

        switch (cellIndex) {
          case 1:
          case 14: {
            const p1 = lerpBottom();
            const p2 = lerpLeft();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            break;
          }
          case 2:
          case 13: {
            const p1 = lerpBottom();
            const p2 = lerpRight();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            break;
          }
          case 3:
          case 12: {
            const p1 = lerpLeft();
            const p2 = lerpRight();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            break;
          }
          case 4:
          case 11: {
            const p1 = lerpRight();
            const p2 = lerpTop();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            break;
          }
          case 5: {
            const p1 = lerpBottom();
            const p2 = lerpRight();
            const p3 = lerpTop();
            const p4 = lerpLeft();
            segments.push({ x1: p1.x, y1: p1.y, x2: p4.x, y2: p4.y });
            segments.push({ x1: p2.x, y1: p2.y, x2: p3.x, y2: p3.y });
            break;
          }
          case 10: {
            const p1 = lerpBottom();
            const p2 = lerpLeft();
            const p3 = lerpTop();
            const p4 = lerpRight();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            segments.push({ x1: p4.x, y1: p4.y, x2: p3.x, y2: p3.y });
            break;
          }
          case 6:
          case 9: {
            const p1 = lerpBottom();
            const p2 = lerpTop();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            break;
          }
          case 7:
          case 8: {
            const p1 = lerpLeft();
            const p2 = lerpTop();
            segments.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
            break;
          }
        }
      }
    }
    result.set(level, segments);
  }

  return result;
}
