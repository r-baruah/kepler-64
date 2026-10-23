//! The v1 laws. Each function implements one section of `02_LAWS_SPEC.md`.

pub mod field;

use crate::params::{Laws, Params};
use crate::position::Position;
use crate::scalar::Scalar;
use cozy_chess::{BitBoard, Color, Piece, Square, get_king_moves};
use field::Cone;

/// L1: rest mass `μ(t)` in scalar type `S` (kings have none).
#[inline]
pub fn rest_mass<S: Scalar>(piece: Piece, l: &Laws<S>) -> S {
    match piece {
        Piece::Pawn => S::cst(1.0),
        Piece::Knight => l.m_n,
        Piece::Bishop => l.m_b,
        Piece::Rook => l.m_r,
        Piece::Queen => l.m_q,
        Piece::King => S::cst(0.0),
    }
}

/// L1 + L6: emission mass `E_p` (carried mass; kings emit `kappa + A_K`).
#[inline]
pub fn emission<S: Scalar>(piece: Piece, acc: f32, l: &Laws<S>) -> S {
    match piece {
        Piece::King => l.kappa + S::cst(acc),
        _ => rest_mass(piece, l) + S::cst(acc),
    }
}

/// L1 material term (rest mass only), White's point of view.
pub fn material<S: Scalar>(pos: &Position, l: &Laws<S>) -> S {
    let b = &pos.board;
    let mut t = S::cst(0.0);
    for piece in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ] {
        let w = b.colored_pieces(Color::White, piece).len() as f32;
        let k = b.colored_pieces(Color::Black, piece).len() as f32;
        t += rest_mass(piece, l) * S::cst(w - k);
    }
    t
}

/// L2 + L3: side fields `Φ_W`, `Φ_B` (index 0 = White).
pub fn side_fields<S: Scalar>(pos: &Position, cones: &[Cone], l: &Laws<S>) -> [[S; 64]; 2] {
    let mut phi = [[S::cst(0.0); 64]; 2];
    let mut alpha_pow = [S::cst(1.0); 3];
    alpha_pow[1] = l.alpha;
    alpha_pow[2] = l.alpha * l.alpha;
    for cone in cones {
        let e = emission(cone.piece, pos.acc[cone.sq as usize], l);
        let side = cone.color as usize;
        for (h, &a_pow) in alpha_pow.iter().enumerate() {
            let n = cone.shell_size(h);
            if n == 0 {
                continue;
            }
            let w = e * a_pow / S::pow_of(n as f32, l.sigma);
            for s in cone.hops[h] {
                phi[side][s as usize] += w;
            }
        }
    }
    phi
}

/// L4a: binding energy term `lambda_bind · (U_B − U_W)`.
pub fn binding<S: Scalar>(pos: &Position, phi: &[[S; 64]; 2], l: &Laws<S>) -> (S, [S; 2]) {
    let b = &pos.board;
    let mut u = [S::cst(0.0); 2];
    for color in [Color::White, Color::Black] {
        let me = color as usize;
        let them = (!color) as usize;
        let bodies = b.colors(color) & !b.pieces(Piece::King);
        for sq in bodies {
            let piece = b.piece_on(sq).expect("occupied");
            let m = rest_mass(piece, l) + S::cst(pos.acc[sq as usize]);
            u[me] += m * (phi[them][sq as usize] - phi[me][sq as usize]);
        }
    }
    (l.lambda_bind * (u[1] - u[0]), u)
}

/// L4b: held space `lambda_space · Σ tanh(Φ_W − Φ_B)`.
pub fn space<S: Scalar>(phi: &[[S; 64]; 2], l: &Laws<S>) -> S {
    let mut t = S::cst(0.0);
    for (w, b) in phi[0].iter().zip(phi[1].iter()) {
        t += (*w - *b).tanh();
    }
    l.lambda_space * t
}

/// One king's Roche quantities (`02_LAWS_SPEC.md` §6).
#[derive(Clone, Copy, Debug)]
pub struct Roche<S> {
    pub square: Square,
    pub body: BitBoard,
    pub stress: S,
    pub binding: S,
    pub eta: S,
    pub gauge: S,
}

pub fn roche<S: Scalar>(pos: &Position, color: Color, phi: &[[S; 64]; 2], l: &Laws<S>) -> Roche<S> {
    let k = pos.board.king(color);
    let body = k.bitboard() | get_king_moves(k);
    let me = color as usize;
    let them = (!color) as usize;
    let mut stress = S::cst(0.0);
    let mut bind = S::cst(1e-3);
    for s in body {
        stress += phi[them][s as usize];
        bind += phi[me][s as usize];
    }
    let eta = stress / bind;
    let gauge = (l.k_roche * (eta - l.rho)).sigmoid();
    Roche {
        square: k,
        body,
        stress,
        binding: bind,
        eta,
        gauge,
    }
}

/// L5 term `beta · (g_black_king − g_white_king)`.
pub fn roche_term<S: Scalar>(w: &Roche<S>, b: &Roche<S>, l: &Laws<S>) -> S {
    l.beta * (b.gauge - w.gauge)
}

/// Convenience used by tests: the constants as f32 laws.
pub fn laws_f32(p: &Params) -> Laws<f32> {
    p.laws::<f32>()
}
