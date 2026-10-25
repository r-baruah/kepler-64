//! Kepler-64 v1 core.
//!
//! The laws are specified in `docs/v1/02_LAWS_SPEC.md`; this crate follows it
//! exactly. Rules (legality, check, mate, draws) come from `cozy-chess`; the
//! evaluation comes only from `laws`.

pub mod clock;
pub mod eval;
pub mod laws;
pub mod params;
pub mod position;
pub mod scalar;
pub mod search;

pub use cozy_chess;
pub use eval::{Breakdown, KingReport, Report, eval_f32, evaluate, report};
pub use params::Params;
pub use position::Position;
pub use scalar::{Dual, NGRAD, Scalar};
