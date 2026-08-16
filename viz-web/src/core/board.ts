/**
 * Kepler-64 Board Representation & Mass Vector Generator
 */

import { PIECE_MASSES } from './constants';

export interface Piece {
  type: 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
  color: 'w' | 'b';
  mass: number; // Base or accreted mass
}

export type BoardState = (Piece | null)[];

export class KeplerBoard {
  public squares: BoardState = new Array(64).fill(null);
  public turn: 'w' | 'b' = 'w';
  public castling = 'KQkq';
  public epSquare: string | null = null;
  public halfmove = 0;
  public fullmove = 1;
  public massBoost: Float32Array = new Float32Array(64);

  constructor(fen?: string) {
    if (fen) {
      this.loadFen(fen);
    }
  }

  public loadFen(fen: string): void {
    this.squares = new Array(64).fill(null);
    this.massBoost = new Float32Array(64);
    const parts = fen.trim().split(/\s+/);
    const pos = parts[0];
    this.turn = (parts[1] === 'b' ? 'b' : 'w');
    this.castling = parts[2] || '-';
    this.epSquare = parts[3] !== '-' ? parts[3] : null;
    this.halfmove = parseInt(parts[4] || '0', 10);
    this.fullmove = parseInt(parts[5] || '1', 10);

    const rows = pos.split('/');
    // In FEN, rows start from rank 8 (top) to rank 1 (bottom).
    // Our internal index: sq = rank * 8 + file, where rank 0 is 1st rank.
    for (let r = 0; r < 8; r++) {
      const fenRank = 7 - r;
      const rowStr = rows[r];
      let file = 0;
      for (let i = 0; i < rowStr.length; i++) {
        const char = rowStr[i];
        if (char >= '1' && char <= '8') {
          file += parseInt(char, 10);
        } else {
          const isWhite = (char === char.toUpperCase());
          const type = char.toLowerCase() as Piece['type'];
          const sq = fenRank * 8 + file;
          const baseMass = PIECE_MASSES[type] || 1.0;
          this.squares[sq] = {
            type,
            color: isWhite ? 'w' : 'b',
            mass: baseMass,
          };
          file++;
        }
      }
    }
  }

  public toFen(): string {
    const rows: string[] = [];
    for (let r = 7; r >= 0; r--) {
      let rowStr = '';
      let emptyCount = 0;
      for (let f = 0; f < 8; f++) {
        const sq = r * 8 + f;
        const p = this.squares[sq];
        if (!p) {
          emptyCount++;
        } else {
          if (emptyCount > 0) {
            rowStr += emptyCount.toString();
            emptyCount = 0;
          }
          const char = (p.color === 'w' ? p.type.toUpperCase() : p.type.toLowerCase());
          rowStr += char;
        }
      }
      if (emptyCount > 0) {
        rowStr += emptyCount.toString();
      }
      rows.push(rowStr);
    }
    return `${rows.join('/')} ${this.turn} ${this.castling} ${this.epSquare || '-'} ${this.halfmove} ${this.fullmove}`;
  }

  /**
   * Signed 64-element mass vector: White pieces > 0, Black pieces < 0.
   */
  public massVector(): Float32Array {
    const mv = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      const p = this.squares[i];
      if (p) {
        mv[i] = (p.color === 'w' ? p.mass + this.massBoost[i] : -(p.mass + this.massBoost[i]));
      } else {
        mv[i] = 0;
      }
    }
    return mv;
  }

  public findKingSquare(color: 'w' | 'b'): number | null {
    for (let i = 0; i < 64; i++) {
      const p = this.squares[i];
      if (p && p.type === 'k' && p.color === color) {
        return i;
      }
    }
    // Fallback: check mass around 1000 with matching sign
    for (let i = 0; i < 64; i++) {
      const p = this.squares[i];
      if (p && Math.abs(p.mass - 1000.0) <= 5.0 && (p.color === color)) {
        return i;
      }
    }
    return null;
  }
}
