/**
 * KaTeX Math Rendering Utility for Kepler-64 Equations
 */

import katex from 'katex';

export function latex(math: string, displayMode: boolean = false): string {
  try {
    return katex.renderToString(math, {
      throwOnError: false,
      displayMode,
    });
  } catch (err) {
    console.error('KaTeX error:', err);
    return math;
  }
}
