//! Number types the evaluation is generic over.
//!
//! Play uses `f32`. Training uses `Dual`: a value plus the exact partial
//! derivatives with respect to the analytically tuned constants
//! (`02_LAWS_SPEC.md` §1, column "grad" = A), computed by forward-mode
//! differentiation through the same code path the engine plays with.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

// Transcendental functions come from the pure-Rust `libm` crate, never the
// platform C library: results must be bit-identical on every machine and in
// WASM (docs/v1/08_DISTRIBUTED_COMPUTE.md, determinism contract).
use libm::{expf, logf, powf, tanhf};

/// Number of analytically tuned constants (see `Params::GRAD_NAMES`).
pub const NGRAD: usize = 13;

pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    fn cst(x: f32) -> Self;
    fn val(&self) -> f32;
    fn tanh(self) -> Self;
    fn sigmoid(self) -> Self;
    /// `n^self` for a constant base `n > 0`.
    fn pow_of(n: f32, e: Self) -> Self;
    /// A seeded variable: value `x`, derivative 1 in slot `i` (ignored by f32).
    fn var(x: f32, i: usize) -> Self;
}

impl Scalar for f32 {
    #[inline(always)]
    fn cst(x: f32) -> Self {
        x
    }
    #[inline(always)]
    fn val(&self) -> f32 {
        *self
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        tanhf(self)
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + expf(-self))
    }
    #[inline(always)]
    fn pow_of(n: f32, e: Self) -> Self {
        powf(n, e)
    }
    #[inline(always)]
    fn var(x: f32, _i: usize) -> Self {
        x
    }
}

/// Forward-mode dual number with `NGRAD` partials.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub v: f32,
    pub d: [f32; NGRAD],
}

impl Dual {
    #[inline(always)]
    fn map(self, v: f32, dv: f32) -> Dual {
        let mut d = self.d;
        for x in d.iter_mut() {
            *x *= dv;
        }
        Dual { v, d }
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline(always)]
    fn add(mut self, o: Dual) -> Dual {
        self.v += o.v;
        for i in 0..NGRAD {
            self.d[i] += o.d[i];
        }
        self
    }
}

impl AddAssign for Dual {
    #[inline(always)]
    fn add_assign(&mut self, o: Dual) {
        *self = *self + o;
    }
}

impl Sub for Dual {
    type Output = Dual;
    #[inline(always)]
    fn sub(self, o: Dual) -> Dual {
        self + (-o)
    }
}

impl Neg for Dual {
    type Output = Dual;
    #[inline(always)]
    fn neg(self) -> Dual {
        self.map(-self.v, -1.0)
    }
}

impl Mul for Dual {
    type Output = Dual;
    #[inline(always)]
    fn mul(self, o: Dual) -> Dual {
        let mut d = [0.0; NGRAD];
        for (i, x) in d.iter_mut().enumerate() {
            *x = self.d[i] * o.v + self.v * o.d[i];
        }
        Dual { v: self.v * o.v, d }
    }
}

impl Div for Dual {
    type Output = Dual;
    #[inline(always)]
    fn div(self, o: Dual) -> Dual {
        let inv = 1.0 / o.v;
        let mut d = [0.0; NGRAD];
        for (i, x) in d.iter_mut().enumerate() {
            *x = (self.d[i] * o.v - self.v * o.d[i]) * inv * inv;
        }
        Dual { v: self.v * inv, d }
    }
}

impl Scalar for Dual {
    #[inline(always)]
    fn cst(x: f32) -> Self {
        Dual {
            v: x,
            d: [0.0; NGRAD],
        }
    }
    #[inline(always)]
    fn val(&self) -> f32 {
        self.v
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        let t = tanhf(self.v);
        self.map(t, 1.0 - t * t)
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        let s = 1.0 / (1.0 + expf(-self.v));
        self.map(s, s * (1.0 - s))
    }
    #[inline(always)]
    fn pow_of(n: f32, e: Self) -> Self {
        let p = powf(n, e.v);
        e.map(p, p * logf(n))
    }
    #[inline(always)]
    fn var(x: f32, i: usize) -> Self {
        let mut d = [0.0; NGRAD];
        d[i] = 1.0;
        Dual { v: x, d }
    }
}
