//! L2 rule-shaped space and L3 light speed (`02_LAWS_SPEC.md` §3–§4).
//!
//! The geometry part (which squares a piece's pull reaches, after how many
//! moves) depends only on the board and `c`, never on the learned constants,
//! so it is computed once in plain bitboards and shared by every scalar type.

use cozy_chess::{
    BitBoard, Board, Color, Piece, Square, get_bishop_moves, get_king_moves, get_knight_moves,
    get_pawn_attacks, get_pawn_quiets, get_rook_moves,
};

/// The squares a piece attacks under the rules: sliders stop at, and include,
/// the first occupied square. No x-rays. Pawn pushes are not attacks.
#[inline]
pub fn attacks(piece: Piece, color: Color, sq: Square, occ: BitBoard) -> BitBoard {
    match piece {
        Piece::Pawn => get_pawn_attacks(sq, color),
        Piece::Knight => get_knight_moves(sq),
        Piece::Bishop => get_bishop_moves(sq, occ),
        Piece::Rook => get_rook_moves(sq, occ),
        Piece::Queen => get_bishop_moves(sq, occ) | get_rook_moves(sq, occ),
        Piece::King => get_king_moves(sq),
    }
}

/// Empty squares the piece could move to (quiet relocations), others frozen.
#[inline]
pub fn relocations(piece: Piece, color: Color, sq: Square, occ: BitBoard) -> BitBoard {
    match piece {
        Piece::Pawn => get_pawn_quiets(sq, color, occ),
        _ => attacks(piece, color, sq, occ) & !occ,
    }
}

/// One piece's light cone: the squares at hop distance 1..=c.
#[derive(Clone, Copy, Debug)]
pub struct Cone {
    pub sq: Square,
    pub piece: Piece,
    pub color: Color,
    /// `hops[h-1]` = squares at hop distance exactly h (empty beyond `c`).
    pub hops: [BitBoard; 3],
}

impl Cone {
    pub fn shell_size(&self, h: usize) -> u32 {
        self.hops[h].len()
    }
}

const NOT_A: u64 = 0xfefe_fefe_fefe_fefe;
const NOT_H: u64 = 0x7f7f_7f7f_7f7f_7f7f;

const NOT_AB: u64 = 0xfcfc_fcfc_fcfc_fcfc;
const NOT_GH: u64 = 0x3f3f_3f3f_3f3f_3f3f;

/// Union of knight moves from every square of `r`.
#[inline(always)]
fn knight_shift(r: u64) -> u64 {
    ((r << 17) & NOT_A)
        | ((r << 15) & NOT_H)
        | ((r << 10) & NOT_AB)
        | ((r << 6) & NOT_GH)
        | ((r >> 17) & NOT_H)
        | ((r >> 15) & NOT_A)
        | ((r >> 10) & NOT_GH)
        | ((r >> 6) & NOT_AB)
}

/// Union of king moves from every square of `r`.
#[inline(always)]
fn king_shift(r: u64) -> u64 {
    (r << 8)
        | (r >> 8)
        | ((r << 1) & NOT_A)
        | ((r >> 1) & NOT_H)
        | ((r << 9) & NOT_A)
        | ((r << 7) & NOT_H)
        | ((r >> 7) & NOT_A)
        | ((r >> 9) & NOT_H)
}

/// Squares one step beyond the empty-square flood of `r` in one direction, i.e.
/// the union of the rays leaving every square of `r` (each ray runs through empty
/// squares and ends on the first occupied one). Kogge-Stone fill. `wrap` masks the
/// squares a step in this direction may enter (file wrap-around).
#[inline(always)]
fn rays<const SH: u32, const LEFT: bool>(r: u64, empty: u64, wrap: u64) -> u64 {
    #[inline(always)]
    fn sh<const LEFT: bool>(x: u64, n: u32) -> u64 {
        if LEFT { x << n } else { x >> n }
    }
    let mut pro = empty & wrap;
    let mut fl = r;
    fl |= pro & sh::<LEFT>(fl, SH);
    pro &= sh::<LEFT>(pro, SH);
    fl |= pro & sh::<LEFT>(fl, 2 * SH);
    pro &= sh::<LEFT>(pro, 2 * SH);
    fl |= pro & sh::<LEFT>(fl, 4 * SH);
    sh::<LEFT>(fl, SH) & wrap
}

/// Hop-2 squares of a slider, before removing hop 1 and the piece's own square.
///
/// The slow definition lifts the piece and slides from every relocation square
/// `r`. Along the line through the piece and `r`, everything that slide reaches
/// is already hop 1 or the piece's own square; across that line (the other
/// directions) the lifted piece is never on the way. So the union of all those
/// slides equals the union of plain rays from the relocation squares on the
/// original board, up to squares that the caller removes anyway. All
/// relocation squares are handled at once by flood fill instead of one magic
/// lookup each. `tests::fast_hops_match_reference` checks this against the slow
/// definition.
#[inline(always)]
fn slider_hop2(piece: Piece, reloc: BitBoard, occ: BitBoard) -> BitBoard {
    let r = reloc.0;
    let empty = !occ.0;
    let mut out = 0u64;
    if piece != Piece::Rook {
        out |= rays::<9, true>(r, empty, NOT_A);
        out |= rays::<7, true>(r, empty, NOT_H);
        out |= rays::<7, false>(r, empty, NOT_A);
        out |= rays::<9, false>(r, empty, NOT_H);
    }
    if piece != Piece::Bishop {
        out |= rays::<8, true>(r, empty, !0);
        out |= rays::<8, false>(r, empty, !0);
        out |= rays::<1, true>(r, empty, NOT_A);
        out |= rays::<1, false>(r, empty, NOT_H);
    }
    BitBoard(out)
}

/// The hop shells of one piece: `[h1, h2, h3]` (empty beyond `c`). Allocation-free.
#[inline(always)]
pub fn hops_of(piece: Piece, color: Color, sq: Square, occ: BitBoard, c: u8) -> [BitBoard; 3] {
    let own = sq.bitboard();
    let h1 = attacks(piece, color, sq, occ) & !own;
    let mut hops = [h1, BitBoard::EMPTY, BitBoard::EMPTY];
    if c >= 2 {
        let lifted = occ & !own;
        let reloc = relocations(piece, color, sq, occ);
        let raw = match piece {
            Piece::Bishop | Piece::Rook | Piece::Queen => slider_hop2(piece, reloc, occ),
            // Attacks of these pieces do not depend on occupancy, so the union
            // over the relocation squares is one set of shifts.
            Piece::Knight => BitBoard(knight_shift(reloc.0)),
            Piece::King => BitBoard(king_shift(reloc.0)),
            Piece::Pawn => BitBoard(match color {
                Color::White => ((reloc.0 << 9) & NOT_A) | ((reloc.0 << 7) & NOT_H),
                Color::Black => ((reloc.0 >> 7) & NOT_A) | ((reloc.0 >> 9) & NOT_H),
            }),
        };
        let h2 = raw & !h1 & !own;
        hops[1] = h2;
        if c >= 3 {
            let mut h3 = BitBoard::EMPTY;
            for r in reloc {
                let occ2 = lifted | r.bitboard();
                let lifted2 = occ2 & !r.bitboard();
                for r2 in relocations(piece, color, r, occ2) {
                    let occ3 = lifted2 | r2.bitboard();
                    h3 |= attacks(piece, color, r2, occ3);
                }
            }
            hops[2] = h3 & !h1 & !h2 & !own;
        }
    }
    hops
}

/// The original definition (`02_LAWS_SPEC.md` §4), one slide per relocation square.
#[cfg(test)]
fn hops_of_reference(
    piece: Piece,
    color: Color,
    sq: Square,
    occ: BitBoard,
    c: u8,
) -> [BitBoard; 3] {
    let own = sq.bitboard();
    let h1 = attacks(piece, color, sq, occ) & !own;
    let mut hops = [h1, BitBoard::EMPTY, BitBoard::EMPTY];
    if c >= 2 {
        let lifted = occ & !own;
        let mut h2 = BitBoard::EMPTY;
        for r in relocations(piece, color, sq, occ) {
            h2 |= attacks(piece, color, r, lifted | r.bitboard());
        }
        let h2 = h2 & !h1 & !own;
        hops[1] = h2;
        if c >= 3 {
            let mut h3 = BitBoard::EMPTY;
            for r in relocations(piece, color, sq, occ) {
                let occ2 = lifted | r.bitboard();
                let lifted2 = occ2 & !r.bitboard();
                for r2 in relocations(piece, color, r, occ2) {
                    h3 |= attacks(piece, color, r2, lifted2 | r2.bitboard());
                }
            }
            hops[2] = h3 & !h1 & !h2 & !own;
        }
    }
    hops
}

pub fn cone(piece: Piece, color: Color, sq: Square, occ: BitBoard, c: u8) -> Cone {
    Cone {
        sq,
        piece,
        color,
        hops: hops_of(piece, color, sq, occ, c),
    }
}

/// Light cones of every piece on the board.
pub fn cones(board: &Board, c: u8) -> Vec<Cone> {
    let occ = board.occupied();
    let mut out = Vec::with_capacity(32);
    for color in [Color::White, Color::Black] {
        for piece in Piece::ALL {
            for sq in board.colored_pieces(color, piece) {
                out.push(cone(piece, color, sq, occ, c));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_hops_match_reference() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..8_000 {
            // Sparse to dense random occupancies.
            let a = next();
            let occ = match next() % 3 {
                0 => a & next() & next(),
                1 => a & next(),
                _ => a,
            };
            for sq in Square::ALL {
                let occ = BitBoard(occ) | sq.bitboard();
                for piece in Piece::ALL {
                    for color in [Color::White, Color::Black] {
                        if piece == Piece::Pawn
                            && matches!(
                                sq.rank(),
                                cozy_chess::Rank::First | cozy_chess::Rank::Eighth
                            )
                        {
                            continue;
                        }
                        for c in [2u8, 3] {
                            let got = hops_of(piece, color, sq, occ, c);
                            let want = hops_of_reference(piece, color, sq, occ, c);
                            assert_eq!(
                                got, want,
                                "{piece:?} {color:?} {sq} occ {:#x} c {c}",
                                occ.0
                            );
                        }
                    }
                }
            }
        }
    }
}
