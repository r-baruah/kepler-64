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
    let s = evaluate::<f32>(pos, p).score_white;
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
