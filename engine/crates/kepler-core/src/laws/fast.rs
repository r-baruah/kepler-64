//! Lean `f32` evaluation path: the same operations in the same order as
//! `eval::evaluate::<f32>`, without building a `Breakdown` (no cone list, no
//! per-evaluation allocation, `n^sigma` from a table). Bit-equality with the
//! generic path is enforced by `tests/eval_golden.rs` and `tests/laws.rs`.

use super::field::hops_of;
use super::{emission, material, rest_mass, roche, roche_term};
use crate::params::{Laws, Params};
use crate::position::Position;
use cozy_chess::{Color, Piece};
use std::cell::RefCell;

/// Largest shell size: at most 63 squares can be reached.
const POW_N: usize = 65;

struct PowTable {
    sigma_bits: u32,
    valid: bool,
    t: [f32; POW_N],
}

thread_local! {
    static POW: RefCell<PowTable> = const {
        RefCell::new(PowTable { sigma_bits: 0, valid: false, t: [0.0; POW_N] })
    };
}

/// `powf(n, sigma)` for n in 0..65, exactly as `Scalar::pow_of` computes it.
#[inline]
fn with_pow_table<R>(sigma: f32, f: impl FnOnce(&[f32; POW_N]) -> R) -> R {
    POW.with(|cell| {
        let mut t = cell.borrow_mut();
        if !t.valid || t.sigma_bits != sigma.to_bits() {
            for n in 1..POW_N {
                t.t[n] = <f32 as crate::scalar::Scalar>::pow_of(n as f32, sigma);
            }
            t.sigma_bits = sigma.to_bits();
            t.valid = true;
        }
        f(&t.t)
    })
}

/// `laws::binding` for f32 (same visit order: colour, then squares ascending),
/// with each body's rest mass taken from the table filled while visiting pieces.
#[inline(always)]
fn binding_f32(pos: &Position, mass: &[f32; 64], phi: &[[f32; 64]; 2], l: &Laws<f32>) -> f32 {
    let b = &pos.board;
    let mut u = [0.0f32; 2];
    for color in [Color::White, Color::Black] {
        let me = color as usize;
        let them = (!color) as usize;
        let mut m = (b.colors(color) & !b.pieces(Piece::King)).0;
        while m != 0 {
            let i = (m.trailing_zeros() as usize) & 63;
            m &= m - 1;
            let mm = mass[i] + pos.acc[i];
            u[me] += mm * (phi[them][i] - phi[me][i]);
        }
    }
    l.lambda_bind * (u[1] - u[0])
}

/// `space` for f32: the same `lambda_space * sum(tanh(w - b))`, summed in the
/// same order; the 64 `tanhf` values come from the batched exact version.
#[inline(always)]
fn space_f32(phi: &[[f32; 64]; 2], l: &Laws<f32>) -> f32 {
    let mut xs = [0.0f32; 64];
    for (x, (w, b)) in xs.iter_mut().zip(phi[0].iter().zip(phi[1].iter())) {
        *x = *w - *b;
    }
    let mut ys = [0.0f32; 64];
    super::tanh::tanh64(&xs, &mut ys);
    let mut t = 0.0f32;
    for y in ys {
        t += y;
    }
    l.lambda_space * t
}

/// `eval::evaluate::<f32>(pos, p).score_white`, computed lean.
///
/// On x86-64 CPUs with AVX2, POPCNT and BMI1 the same code runs compiled with
/// those instructions (hardware popcount, bit scan, wider vectors). No FMA is
/// enabled, so every floating-point result is bit-identical to the baseline
/// build; only integer and vector-width instructions change.
pub fn score_white_f32(pos: &Position, p: &Params) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("bmi1")
        {
            // SAFETY: the required CPU features were just detected.
            return unsafe { score_white_f32_avx2(pos, p) };
        }
    }
    score_white_f32_impl(pos, p)
}

/// The baseline-instruction build of the same function, for tests that check
/// the accelerated build against it.
pub fn score_white_f32_portable(pos: &Position, p: &Params) -> f32 {
    score_white_f32_impl(pos, p)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt,bmi1")]
unsafe fn score_white_f32_avx2(pos: &Position, p: &Params) -> f32 {
    score_white_f32_impl(pos, p)
}

#[inline(always)]
fn score_white_f32_impl(pos: &Position, p: &Params) -> f32 {
    let l: Laws<f32> = p.laws::<f32>();
    let b = &pos.board;
    let occ = b.occupied();
    let mut phi = [[0.0f32; 64]; 2];
    let mut mass = [0.0f32; 64];
    let alpha_pow = [1.0f32, l.alpha, l.alpha * l.alpha];
    with_pow_table(l.sigma, |pw| {
        // Same visit order as `field::cones`: colour, piece type, square.
        for color in [Color::White, Color::Black] {
            let side = color as usize;
            macro_rules! pass {
                ($piece:expr) => {
                    for sq in b.colored_pieces(color, $piece) {
                        let hops = hops_of($piece, color, sq, occ, p.c);
                        let e = emission($piece, pos.acc[sq as usize], &l);
                        mass[sq as usize] = rest_mass($piece, &l);
                        for (h, &a_pow) in alpha_pow.iter().enumerate() {
                            let n = hops[h].len() as usize;
                            if n == 0 {
                                continue;
                            }
                            let w = e * a_pow / pw[n];
                            let mut m = hops[h].0;
                            while m != 0 {
                                let i = m.trailing_zeros() as usize;
                                m &= m - 1;
                                phi[side][i & 63] += w;
                            }
                        }
                    }
                };
            }
            pass!(Piece::Pawn);
            pass!(Piece::Knight);
            pass!(Piece::Bishop);
            pass!(Piece::Rook);
            pass!(Piece::Queen);
            pass!(Piece::King);
        }
    });
    let mat = material(pos, &l);
    let bind = binding_f32(pos, &mass, &phi, &l);
    let sp = space_f32(&phi, &l);
    let kw = roche(pos, Color::White, &phi, &l);
    let kb = roche(pos, Color::Black, &phi, &l);
    let ro = roche_term(&kw, &kb, &l);
    mat + bind + sp + ro
}
