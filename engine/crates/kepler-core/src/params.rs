//! The 15 constants of the v1 universe (`02_LAWS_SPEC.md` §1).

use crate::scalar::Scalar;
use serde::{Deserialize, Serialize};

/// Spec id this code implements. Bump together with `02_LAWS_SPEC.md` §12.
pub const SPEC_ID: &str = "v1.0-spec-2";
/// The spec-1 id. A universe with `lambda_tide = 0` is exactly a spec-1 universe,
/// so files describing one keep this id (`02_LAWS_SPEC.md` §12).
pub const SPEC_ID_1: &str = "v1.0-spec-1";

/// The spec id a file describing `p` is written with: spec-1 when `lambda_tide`
/// is 0 (the field is then omitted too), spec-2 otherwise.
pub fn spec_id(p: &Params) -> &'static str {
    if p.lambda_tide == 0.0 {
        SPEC_ID_1
    } else {
        SPEC_ID
    }
}

/// The spec id for a file that holds two universes (a match or SPRT shard):
/// spec-2 if either one uses `lambda_tide`.
pub fn spec_id_pair(a: &Params, b: &Params) -> &'static str {
    if a.lambda_tide == 0.0 && b.lambda_tide == 0.0 {
        SPEC_ID_1
    } else {
        SPEC_ID
    }
}

/// Whether `s` is a spec id this code can read.
pub fn is_known_spec(s: &str) -> bool {
    s == SPEC_ID || s == SPEC_ID_1
}

fn is_zero(x: &f32) -> bool {
    *x == 0.0
}

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
    /// spec-2 constant 15. Omitted from JSON when 0, so spec-1 files are unchanged.
    #[serde(skip_serializing_if = "is_zero")]
    pub lambda_tide: f32,
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
            lambda_tide: 0.0,
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
    pub lambda_tide: S,
}

impl Params {
    /// Analytically tuned constants, in gradient-slot order.
    pub const GRAD_NAMES: [&'static str; 13] = [
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
        "lambda_tide",
    ];

    /// Bounds per gradient slot (`02_LAWS_SPEC.md` §1).
    pub const GRAD_BOUNDS: [(f32, f32); 13] = [
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
        (0.0, 2.0),
    ];

    pub fn grad_values(&self) -> [f32; 13] {
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
            self.lambda_tide,
        ]
    }

    /// Set the gradient-slot values, projected onto their bounds.
    pub fn set_grad_values(&mut self, v: [f32; 13]) {
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
        self.lambda_tide = p(12);
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
            lambda_tide: S::var(v[12], 12),
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
        if g.spec != SPEC_ID && g.spec != SPEC_ID_1 {
            return Err(format!(
                "generation spec {} is not one of {SPEC_ID_1}, {SPEC_ID}",
                g.spec
            ));
        }
        if g.spec == SPEC_ID_1 && g.constants.lambda_tide != 0.0 {
            return Err(format!(
                "a {SPEC_ID_1} file cannot set lambda_tide ({}); that constant is {SPEC_ID}",
                g.constants.lambda_tide
            ));
        }
        g.constants.validate()?;
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_json(spec: &str, constants: &str) -> String {
        format!("{{\"generation\": 1, \"spec\": \"{spec}\", \"constants\": {constants}}}")
    }

    #[test]
    fn lambda_tide_defaults_to_zero_and_is_omitted() {
        let p: Params = serde_json::from_str("{\"rho\": 2.0}").unwrap();
        assert_eq!(p.lambda_tide, 0.0);
        assert!(!serde_json::to_string(&p).unwrap().contains("lambda_tide"));
        let q = Params {
            lambda_tide: 0.34,
            ..p
        };
        assert!(serde_json::to_string(&q).unwrap().contains("lambda_tide"));
        assert_eq!(spec_id(&p), SPEC_ID_1);
        assert_eq!(spec_id(&q), SPEC_ID);
    }

    #[test]
    fn spec_ids_and_lambda_tide_rules() {
        assert!(Generation::from_json(&gen_json(SPEC_ID_1, "{}")).is_ok());
        assert!(Generation::from_json(&gen_json(SPEC_ID, "{}")).is_ok());
        assert!(Generation::from_json(&gen_json(SPEC_ID, "{\"lambda_tide\": 0.5}")).is_ok());
        assert!(Generation::from_json(&gen_json(SPEC_ID_1, "{\"lambda_tide\": 0.5}")).is_err());
        assert!(Generation::from_json(&gen_json(SPEC_ID, "{\"lambda_tide\": 2.5}")).is_err());
        assert!(Generation::from_json(&gen_json("v1.0-spec-0", "{}")).is_err());
    }

    #[test]
    fn every_generation_file_loads() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/generations");
        let mut n = 0;
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("gen-") && name.ends_with(".json") {
                let g = Generation::from_json(&std::fs::read_to_string(e.path()).unwrap())
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(g.constants.lambda_tide, 0.0, "{name}");
                n += 1;
            }
        }
        assert!(n >= 4);
    }
}
