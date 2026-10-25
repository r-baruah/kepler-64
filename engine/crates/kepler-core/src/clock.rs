//! `Instant` that also exists on `wasm32-unknown-unknown`, where
//! `std::time::Instant::now()` panics.
//!
//! On native targets this is `std::time::Instant`. On wasm32 it is a stub whose
//! `elapsed()` is always zero, so time limits never fire there; the wasm wrapper
//! turns `movetimeMs` into a node budget instead (`03_ENGINE_ARCHITECTURE.md` §6.2).

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant
    }
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
