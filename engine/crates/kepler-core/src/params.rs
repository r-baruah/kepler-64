//! The 14 constants of the v1 universe (`02_LAWS_SPEC.md` §1).

use crate::scalar::Scalar;
use serde::{Deserialize, Serialize};

/// Spec id this code implements. Bump together with `02_LAWS_SPEC.md` §12.
pub const SPEC_ID: &str = "v1.0-spec-1";

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    pub m_n: f32,
    pub m_b: f32,
    pub m_r: f32,
    pub m_q: f32,
    pub kappa: f32,
    pub alpha: f32,
    pub sigma: f32,
    pub c: u8,
    pub lambda_bind: f32,
    pub lambda_space: f32,
    pub rho: f32,
    pub k_roche: f32,
    pub beta: f32,
    pub f_acc: f32,
}

impl Default for Params {
    /// Gen 0, before calibration. `lambda_bind`, `lambda_space` and `beta`
    /// are provisional until `kepler-train calibrate` writes gen-000.json.
    fn default() -> Self {
        Params {
            m_n: 3.0,
            m_b: 3.0,
            m_r: 5.0,
            m_q: 9.0,
            kappa: 1.0,
            alpha: 0.35,
            sigma: 0.5,
            c: 2,
            lambda_bind: 0.01,
            lambda_space: 0.05,
            rho: 1.5,
            k_roche: 3.0,
            beta: 2.0,
            f_acc: 0.8,
        }
    }
}

/// The laws' continuous constants as the scalar type the evaluation runs in.
#[derive(Clone, Copy, Debug)]
pub struct Laws<S> {
    pub m_n: S,
    pub m_b: S,
    pub m_r: S,
    pub m_q: S,
    pub kappa: S,
    pub alpha: S,
    pub sigma: S,
    pub lambda_bind: S,
    pub lambda_space: S,
    pub rho: S,
    pub k_roche: S,
    pub beta: S,
}

impl Params {
    /// Analytically tuned constants, in gradient-slot order.
    pub const GRAD_NAMES: [&'static str; 12] = [
        "m_n",
        "m_b",
        "m_r",
        "m_q",
        "kappa",
        "alpha",
        "sigma",
        "lambda_bind",
        "lambda_space",
        "rho",
        "k_roche",
        "beta",
    ];

    /// Bounds per gradient slot (`02_LAWS_SPEC.md` §1).
    pub const GRAD_BOUNDS: [(f32, f32); 12] = [
        (1.0, 8.0),
        (1.0, 8.0),
        (2.0, 12.0),
        (4.0, 20.0),
        (0.0, 5.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 2.0),
        (0.1, 10.0),
        (0.1, 20.0),
        (0.0, 30.0),
    ];

    pub fn grad_values(&self) -> [f32; 12] {
        [
            self.m_n,
            self.m_b,
            self.m_r,
            self.m_q,
            self.kappa,
            self.alpha,
            self.sigma,
            self.lambda_bind,
            self.lambda_space,
            self.rho,
            self.k_roche,
            self.beta,
        ]
    }

    /// Set the gradient-slot values, projected onto their bounds.
    pub fn set_grad_values(&mut self, v: [f32; 12]) {
        let p = |i: usize| v[i].clamp(Self::GRAD_BOUNDS[i].0, Self::GRAD_BOUNDS[i].1);
        self.m_n = p(0);
        self.m_b = p(1);
        self.m_r = p(2);
        self.m_q = p(3);
        self.kappa = p(4);
        self.alpha = p(5);
        self.sigma = p(6);
        self.lambda_bind = p(7);
        self.lambda_space = p(8);
        self.rho = p(9);
        self.k_roche = p(10);
        self.beta = p(11);
    }

    /// The continuous constants in scalar type `S`; for `Dual` each one is
    /// seeded in its own gradient slot.
    pub fn laws<S: Scalar>(&self) -> Laws<S> {
        let v = self.grad_values();
        Laws {
            m_n: S::var(v[0], 0),
            m_b: S::var(v[1], 1),
            m_r: S::var(v[2], 2),
            m_q: S::var(v[3], 3),
            kappa: S::var(v[4], 4),
            alpha: S::var(v[5], 5),
            sigma: S::var(v[6], 6),
            lambda_bind: S::var(v[7], 7),
            lambda_space: S::var(v[8], 8),
            rho: S::var(v[9], 9),
            k_roche: S::var(v[10], 10),
            beta: S::var(v[11], 11),
        }
    }

    /// Rest mass of a non-king piece type in f32 (search ordering, accretion).
    pub fn rest_mass(&self, piece: cozy_chess::Piece) -> f32 {
        use cozy_chess::Piece::*;
        match piece {
            Pawn => 1.0,
            Knight => self.m_n,
            Bishop => self.m_b,
            Rook => self.m_r,
            Queen => self.m_q,
            King => 0.0,
        }
    }
}

/// A generation file (`engine/data/generations/gen-XXX.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Generation {
    pub generation: u32,
    #[serde(default)]
    pub parent: Option<u32>,
    pub spec: String,
    #[serde(default)]
    pub created: String,
    pub constants: Params,
    #[serde(default)]
    pub pinned: Vec<String>,
    #[serde(default)]
    pub training: serde_json::Value,
    #[serde(default)]
    pub acceptance: serde_json::Value,
}

impl Params {
    /// Check the constants against `02_LAWS_SPEC.md` §1 bounds.
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=3).contains(&self.c) {
            return Err(format!("c = {} is outside {{1, 2, 3}}", self.c));
        }
        if !(0.0..=1.0).contains(&self.f_acc) {
            return Err(format!("f_acc = {} is outside [0, 1]", self.f_acc));
        }
        for ((name, v), (lo, hi)) in Self::GRAD_NAMES
            .iter()
            .zip(self.grad_values())
            .zip(Self::GRAD_BOUNDS)
        {
            if !(lo..=hi).contains(&v) {
                return Err(format!("{name} = {v} is outside [{lo}, {hi}]"));
            }
        }
        Ok(())
    }
}

impl Generation {
    /// Parse a generation file; the spec id must match and the constants must
    /// be in bounds, so a stale or hand-edited file fails loudly.
    pub fn from_json(s: &str) -> Result<Self, String> {
        let g: Generation = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if g.spec != SPEC_ID {
            return Err(format!(
                "generation spec {} does not match engine spec {SPEC_ID}",
                g.spec
            ));
        }
        g.constants.validate()?;
        Ok(g)
    }
}
