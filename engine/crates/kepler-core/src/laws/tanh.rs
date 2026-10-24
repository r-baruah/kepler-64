//! Batched `tanhf`, bit-identical to `libm::tanhf`.
//!
//! The L4b space term needs 64 `tanhf` calls per evaluation, and `libm`'s version
//! branches on the argument, which the CPU predicts badly. For the argument range
//! that covers about 80% of squares, `libm` always takes the same path
//! (`expm1f` with `k` in 2..=20; the other ranges are handled the same way). That path is reproduced here as straight-line
//! `f32` arithmetic, operation for operation, so the compiler can run it on
//! several squares at once. Every other argument goes to `libm::tanhf` itself.
//! `tests::exhaustive_fast_range` checks all ~30M arguments of the fast range.

#[allow(clippy::excessive_precision)]
const LN2_HI: f32 = 6.9313812256e-01;
#[allow(clippy::excessive_precision)]
const LN2_LO: f32 = 9.0580006145e-06;
#[allow(clippy::excessive_precision)]
const INV_LN2: f32 = 1.4426950216e+00;
#[allow(clippy::excessive_precision)]
const Q1: f32 = -3.3333212137e-2;
#[allow(clippy::excessive_precision)]
const Q2: f32 = 1.5807170421e-3;

/// `|x|` above `log(3)/2` (0.5493): libm computes `1 - 2/(expm1f(2|x|) + 2)`.
const BIG: u32 = 0x3f0c_9f54;
/// `|x|` above `log(5/3)/2` (0.2554): libm computes `t/(t+2)`, `t = expm1f(2|x|)`.
/// This is the lower end of the fast range; smaller arguments use libm itself.
const FAST_LO: u32 = 0x3e82_c578;
/// Above 10 libm returns 1 (`1 + 0/|x|`).
const SAT: u32 = 0x4120_0000;
/// `expm1f` takes its `k = 1` branch below `1.5 ln 2`.
const K1_LIM: u32 = 0x3f85_1592;

/// `tanhf(|x|)` for `|x|` bits in `(FAST_LO, 0x7f800000)`; garbage (but no panic)
/// for any other `a`. Mirrors `libm::tanhf` and `libm::expm1f` operation for
/// operation, with the branches turned into selects so it can run on several
/// values at once.
#[inline(always)]
fn core(a: f32) -> f32 {
    let ab = a.to_bits();
    let a2 = 2. * a;
    // Clamped so the conversion is in range for every input (NaN becomes 0);
    // inside the fast range the clamp never binds up to |x| = 10 and this is `as i32`.
    // `f32::clamp` would pass NaN through, which is not allowed here.
    #[allow(clippy::manual_clamp)]
    let kf = (INV_LN2 * a2 + 0.5).max(0.0).min(100.0);
    // SAFETY: `kf` is in [0, 100].
    let k_gen = unsafe { kf.to_int_unchecked::<i32>() };
    let k = if a2.to_bits() < K1_LIM { 1 } else { k_gen };
    let t = k as f32;
    let hi = a2 - t * LN2_HI;
    let lo = t * LN2_LO;
    let x = hi - lo;
    let c = (hi - x) - lo;
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1. + hxs * (Q1 + hxs * Q2);
    let t3 = 3. - r1 * hfx;
    let mut e = hxs * ((r1 - t3) / (6. - x * t3));
    e = x * (e - c) - c;
    e -= hxs;
    let twopk = f32::from_bits(0x7f_i32.wrapping_add(k).wrapping_shl(23) as u32);
    let uf = f32::from_bits(0x7f_i32.wrapping_sub(k).wrapping_shl(23) as u32);
    let em_k1 = 1. + 2. * (x - e);
    let em_lo = (x - e + (1. - uf)) * twopk;
    let em_hi = (x - (e + uf) + 1.) * twopk;
    let em = if k == 1 {
        em_k1
    } else if k < 23 {
        em_lo
    } else {
        em_hi
    };
    let y_big = 1. - 2. / (em + 2.);
    let y_mid = em / (em + 2.);
    let y = if ab > BIG { y_big } else { y_mid };
    if ab > SAT { 1.0 } else { y }
}

/// `out[i] = libm::tanhf(xs[i])` for all 64 lanes.
#[inline(always)]
pub fn tanh64(xs: &[f32; 64], out: &mut [f32; 64]) {
    for i in 0..64 {
        let a = f32::from_bits(xs[i].to_bits() & 0x7fff_ffff);
        out[i] = core(a);
    }
    for i in 0..64 {
        let bits = xs[i].to_bits();
        let abits = bits & 0x7fff_ffff;
        if abits > FAST_LO && abits < 0x7f80_0000 {
            out[i] = f32::from_bits(out[i].to_bits() ^ (bits & 0x8000_0000));
        } else if abits == 0 {
            // libm returns its argument, with its sign.
            out[i] = xs[i];
        } else {
            out[i] = libm::tanhf(xs[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_fast_range() {
        for bits in (FAST_LO + 1)..=SAT {
            let a = f32::from_bits(bits);
            assert_eq!(core(a).to_bits(), libm::tanhf(a).to_bits(), "a = {a}");
        }
        for bits in ((SAT + 1)..0x7f80_0000).step_by(251) {
            let a = f32::from_bits(bits);
            assert_eq!(core(a).to_bits(), libm::tanhf(a).to_bits(), "a = {a}");
        }
    }

    #[test]
    fn batch_matches_libm_everywhere() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut xs = [0f32; 64];
        let mut out = [0f32; 64];
        for _ in 0..200_000 {
            for x in xs.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Mix of arbitrary bit patterns and values in the field range.
                *x = if s & 1 == 0 {
                    f32::from_bits((s >> 8) as u32)
                } else {
                    ((s >> 16) as f32 / 4_294_967_296.0 - 0.5) * 24.0
                };
                if x.is_nan() {
                    *x = 0.0;
                }
            }
            tanh64(&xs, &mut out);
            for i in 0..64 {
                assert_eq!(
                    out[i].to_bits(),
                    libm::tanhf(xs[i]).to_bits(),
                    "x = {}",
                    xs[i]
                );
            }
        }
    }
}
