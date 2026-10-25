//! A position = the chess board plus the physical state the rules don't carry:
//! the mass each piece has absorbed by capturing (L6, `02_LAWS_SPEC.md` §7).

use crate::params::Params;
use cozy_chess::{Board, Color, File, Move, Piece, Rank, Square};

#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    pub board: Board,
    /// Accreted excess mass per square, moved with the piece standing on it.
    pub acc: [f32; 64],
}

impl Default for Position {
    fn default() -> Self {
        Position::startpos()
    }
}

impl Position {
    pub fn startpos() -> Self {
        Position {
            board: Board::default(),
            acc: [0.0; 64],
        }
    }

    /// From FEN. Accreted mass is unknown from FEN, so it starts at zero.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let board = Board::from_fen(fen, false).map_err(|e| format!("{e:?}"))?;
        Ok(Position {
            board,
            acc: [0.0; 64],
        })
    }

    /// Carried mass `M = μ + A` of the non-king piece on `sq`, if any.
    pub fn carried_mass(&self, sq: Square, p: &Params) -> f32 {
        match self.board.piece_on(sq) {
            Some(Piece::King) | None => 0.0,
            Some(pc) => p.rest_mass(pc) + self.acc[sq as usize],
        }
    }

    /// Play a legal move (cozy-chess encoding: castling is king-takes-own-rook)
    /// and carry accreted mass forward exactly (`02_LAWS_SPEC.md` §7).
    pub fn play(&self, mv: Move, p: &Params) -> Position {
        let b = &self.board;
        let us = b.side_to_move();
        let mut acc = self.acc;
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = b.piece_on(mv.from).expect("move from an empty square");

        if moving == Piece::King && b.color_on(mv.to) == Some(us) {
            // Castling: the king and the rook keep their own accreted mass.
            let rank = Rank::First.relative_to(us);
            let kingside = mv.to.file() > mv.from.file();
            let (kf, rf) = if kingside {
                (File::G, File::F)
            } else {
                (File::C, File::D)
            };
            let (ka, ra) = (acc[from], acc[to]);
            acc[from] = 0.0;
            acc[to] = 0.0;
            acc[Square::new(kf, rank) as usize] = ka;
            acc[Square::new(rf, rank) as usize] = ra;
        } else {
            let mut gained = 0.0;
            let victim_sq = if moving == Piece::Pawn
                && mv.from.file() != mv.to.file()
                && b.piece_on(mv.to).is_none()
            {
                // En passant: the victim stands beside the captor's origin.
                Some(Square::new(mv.to.file(), mv.from.rank()))
            } else if b.color_on(mv.to) == Some(!us) {
                Some(mv.to)
            } else {
                None
            };
            if let Some(v) = victim_sq {
                let vm = match b.piece_on(v) {
                    Some(Piece::King) | None => 0.0,
                    Some(pc) => p.rest_mass(pc) + acc[v as usize],
                };
                gained = p.f_acc * vm;
                acc[v as usize] = 0.0;
            }
            let carried = acc[from] + gained;
            acc[from] = 0.0;
            acc[to] = carried;
        }

        let mut board = b.clone();
        board.play_unchecked(mv);
        Position { board, acc }
    }

    /// Mirror colours and ranks (for the colour-symmetry invariant).
    pub fn mirrored(&self) -> Position {
        let b = &self.board;
        let mut fen_rows = Vec::with_capacity(8);
        for r in (0..8).rev() {
            // Rank r of the mirror is rank 7 - r of the original, colours swapped.
            let src = 7 - r;
            let mut row = String::new();
            let mut empty = 0;
            for f in 0..8 {
                let sq = Square::new(File::index(f), Rank::index(src));
                match (b.piece_on(sq), b.color_on(sq)) {
                    (Some(pc), Some(col)) => {
                        if empty > 0 {
                            row.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let ch: char = pc.into();
                        row.push(if col == Color::White {
                            ch
                        } else {
                            ch.to_ascii_uppercase()
                        });
                    }
                    _ => empty += 1,
                }
            }
            if empty > 0 {
                row.push_str(&empty.to_string());
            }
            fen_rows.push(row);
        }
        let stm = if b.side_to_move() == Color::White {
            "b"
        } else {
            "w"
        };
        let fen = format!("{} {} - - 0 1", fen_rows.join("/"), stm);
        let mut m = Position::from_fen(&fen).expect("mirrored FEN is valid");
        for i in 0..64 {
            let sq = Square::index(i);
            let ms = sq.flip_rank();
            m.acc[ms as usize] = self.acc[i];
        }
        m
    }
}
