//! Total evaluation (`02_LAWS_SPEC.md` §8) and the breakdown it exposes (§10).

use crate::laws::{self, field};
use crate::params::Params;
use crate::position::Position;
use crate::scalar::Scalar;
use cozy_chess::{BitBoard, Color, Square};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct Breakdown<S> {
    pub material: S,
    pub bind: S,
    pub space: S,
    pub roche: S,
    pub score_white: S,
    /// `field[0]` = Φ_W, `field[1]` = Φ_B.
    pub field: [[S; 64]; 2],
    /// Binding energy per side, `U_W`, `U_B`.
    pub energy: [S; 2],
    /// `kings[0]` = White's king, `kings[1]` = Black's.
    pub kings: [laws::Roche<S>; 2],
    pub cones: Vec<field::Cone>,
}

/// The full evaluation in scalar type `S`.
pub fn evaluate<S: Scalar>(pos: &Position, p: &Params) -> Breakdown<S> {
    let l = p.laws::<S>();
    let cones = field::cones(&pos.board, p.c);
    let phi = laws::side_fields(pos, &cones, &l);
    let material = laws::material(pos, &l);
    let (bind, energy) = laws::binding(pos, &phi, &l);
    let space = laws::space(&phi, &l);
    let kw = laws::roche(pos, Color::White, &phi, &l);
    let kb = laws::roche(pos, Color::Black, &phi, &l);
    let roche = laws::roche_term(&kw, &kb, &l);
    let score_white = material + bind + space + roche;
    Breakdown {
        material,
        bind,
        space,
        roche,
        score_white,
        field: phi,
        energy,
        kings: [kw, kb],
        cones,
    }
}

/// Score from the side to move's point of view, in Kepler units (f32 play path).
#[inline]
pub fn eval_f32(pos: &Position, p: &Params) -> f32 {
    let s = laws::fast::score_white_f32(pos, p);
    if pos.board.side_to_move() == Color::White {
        s
    } else {
        -s
    }
}

/// JSON-friendly report of one king, for UCI `info string` and the website.
#[derive(Clone, Debug, Serialize)]
pub struct KingReport {
    pub square: String,
    pub body: Vec<String>,
    pub stress: f32,
    pub binding: f32,
    pub eta: f32,
    pub gauge: f32,
    /// `eta / rho`; 1.0 = the breaking point (shown as 100%).
    pub break_ratio: f32,
}

fn squares(bb: BitBoard) -> Vec<String> {
    bb.into_iter().map(|s: Square| s.to_string()).collect()
}

impl Breakdown<f32> {
    pub fn king_report(&self, color: Color, p: &Params) -> KingReport {
        let k = &self.kings[color as usize];
        KingReport {
            square: k.square.to_string(),
            body: squares(k.body),
            stress: k.stress,
            binding: k.binding,
            eta: k.eta,
            gauge: k.gauge,
            break_ratio: k.eta / p.rho,
        }
    }
}

/// The four terms and the total, White's point of view (`02_LAWS_SPEC.md` §10).
#[derive(Clone, Debug, Serialize)]
pub struct TermsReport {
    pub material: f32,
    pub bind: f32,
    pub space: f32,
    pub roche: f32,
    pub score_white: f32,
}

/// Both side fields, 64 values each, indexed by square (a1 = 0, b1 = 1, ..., h8 = 63).
#[derive(Clone, Debug, Serialize)]
pub struct FieldReport {
    pub white: Vec<f32>,
    pub black: Vec<f32>,
}

/// One piece with its mass bookkeeping and light cone (§10 "per piece").
#[derive(Clone, Debug, Serialize)]
pub struct PieceReport {
    pub square: String,
    /// Piece type as an upper-case letter: P N B R Q K.
    pub piece: char,
    /// "white" or "black".
    pub color: &'static str,
    /// `mu`: rest mass (0 for kings).
    pub rest_mass: f32,
    /// `A`: accreted mass.
    pub accreted: f32,
    /// `M = mu + A` (0 for kings).
    pub carried_mass: f32,
    /// What the piece emits into the field: `M` for pieces, `kappa + A` for kings.
    pub emission: f32,
    /// Binding energy `e_q = M (Phi_enemy - Phi_own)` at its square (0 for kings).
    pub energy: f32,
    /// Squares at hop distance 1, 2 and 3 (its light cone; hop 3 is empty when `c < 3`).
    pub hops: [Vec<String>; 3],
}

/// The full §10 breakdown, ready for JSON.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub terms: TermsReport,
    pub field: FieldReport,
    /// Sorted White first, then Black; within a side by piece type then square.
    pub pieces: Vec<PieceReport>,
    /// White's king, then Black's.
    pub kings: [KingReport; 2],
    pub constants: Params,
    pub generation: Option<u32>,
}

fn piece_letter(p: cozy_chess::Piece) -> char {
    use cozy_chess::Piece::*;
    match p {
        Pawn => 'P',
        Knight => 'N',
        Bishop => 'B',
        Rook => 'R',
        Queen => 'Q',
        King => 'K',
    }
}

/// Build the §10 report. This is an additional view of `evaluate::<f32>`; the
/// evaluation itself is untouched.
pub fn report(pos: &Position, p: &Params, generation: Option<u32>) -> Report {
    let bd = evaluate::<f32>(pos, p);
    let pieces = bd
        .cones
        .iter()
        .map(|cone| {
            let is_king = cone.piece == cozy_chess::Piece::King;
            let acc = pos.acc[cone.sq as usize];
            let rest = p.rest_mass(cone.piece);
            let carried = if is_king { 0.0 } else { rest + acc };
            let me = cone.color as usize;
            let them = 1 - me;
            let sq = cone.sq as usize;
            let energy = if is_king {
                0.0
            } else {
                carried * (bd.field[them][sq] - bd.field[me][sq])
            };
            PieceReport {
                square: cone.sq.to_string(),
                piece: piece_letter(cone.piece),
                color: if cone.color == Color::White {
                    "white"
                } else {
                    "black"
                },
                rest_mass: rest,
                accreted: acc,
                carried_mass: carried,
                emission: if is_king { p.kappa + acc } else { carried },
                energy,
                hops: [
                    squares(cone.hops[0]),
                    squares(cone.hops[1]),
                    squares(cone.hops[2]),
                ],
            }
        })
        .collect();
    Report {
        terms: TermsReport {
            material: bd.material,
            bind: bd.bind,
            space: bd.space,
            roche: bd.roche,
            score_white: bd.score_white,
        },
        field: FieldReport {
            white: bd.field[0].to_vec(),
            black: bd.field[1].to_vec(),
        },
        pieces,
        kings: [
            bd.king_report(Color::White, p),
            bd.king_report(Color::Black, p),
        ],
        constants: *p,
        generation,
    }
}

#[cfg(test)]
mod report_tests {
    use super::*;

    #[test]
    fn report_energy_matches_breakdown() {
        let p = Params {
            lambda_bind: 0.1,
            ..Params::default()
        };
        let mut pos = Position::startpos();
        for uci in ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "g1f3"] {
            let mv = crate::search::move_from_uci(&pos.board, uci).unwrap();
            pos = pos.play(mv, &p);
        }
        for fen in [
            None,
            Some("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ] {
            let pos = fen.map_or(pos.clone(), |f| Position::from_fen(f).unwrap());
            let bd = evaluate::<f32>(&pos, &p);
            let r = report(&pos, &p, Some(7));
            assert_eq!(r.generation, Some(7));
            assert_eq!(r.field.white.len(), 64);
            assert_eq!(r.pieces.len(), pos.board.occupied().len() as usize);
            for (i, color) in ["white", "black"].iter().enumerate() {
                let sum: f32 = r
                    .pieces
                    .iter()
                    .filter(|x| x.color == *color)
                    .map(|x| x.energy)
                    .sum();
                assert!(
                    (sum - bd.energy[i]).abs() < 1e-4,
                    "{color}: {sum} vs {}",
                    bd.energy[i]
                );
            }
            let t = &r.terms;
            assert_eq!(t.material + t.bind + t.space + t.roche, t.score_white);
            assert!(
                r.pieces
                    .iter()
                    .filter(|x| x.piece == 'K')
                    .all(|k| k.energy == 0.0)
            );
            assert!((r.kings[0].break_ratio - bd.kings[0].eta / p.rho).abs() < 1e-6);
            serde_json::to_string(&r).unwrap();
        }
    }
}
