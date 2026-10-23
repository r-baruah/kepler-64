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

pub fn cone(piece: Piece, color: Color, sq: Square, occ: BitBoard, c: u8) -> Cone {
    let own = sq.bitboard();
    let h1 = attacks(piece, color, sq, occ) & !own;
    let mut hops = [h1, BitBoard::EMPTY, BitBoard::EMPTY];
    if c >= 2 {
        let lifted = occ & !own;
        let mut h2 = BitBoard::EMPTY;
        let mut frontier: Vec<(Square, BitBoard)> = Vec::new();
        for r in relocations(piece, color, sq, occ) {
            let occ2 = lifted | r.bitboard();
            h2 |= attacks(piece, color, r, occ2);
            if c >= 3 {
                frontier.push((r, occ2));
            }
        }
        let h2 = h2 & !h1 & !own;
        hops[1] = h2;
        if c >= 3 {
            let mut h3 = BitBoard::EMPTY;
            for (r, occ2) in frontier {
                let lifted2 = occ2 & !r.bitboard();
                for r2 in relocations(piece, color, r, occ2) {
                    let occ3 = lifted2 | r2.bitboard();
                    h3 |= attacks(piece, color, r2, occ3);
                }
            }
            hops[2] = h3 & !h1 & !h2 & !own;
        }
    }
    Cone {
        sq,
        piece,
        color,
        hops,
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
