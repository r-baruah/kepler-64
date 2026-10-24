//! `kepler-wasm`: the `Engine` class of `03_ENGINE_ARCHITECTURE.md` §6.2.
//!
//! The game logic lives in [`Game`], which is plain Rust (testable natively).
//! [`Engine`] is a thin `wasm-bindgen` wrapper that converts to and from JS.
//!
//! Conventions
//! - Moves are UCI strings with standard castling (`e1g1`).
//! - `breakdown()`, `state()` and `search()` return plain JS objects (built with
//!   `serde-wasm-bindgen`, JSON-compatible: structs are objects, no `Map`s).
//! - `movetimeMs` is honoured as a node budget, because `std::time::Instant`
//!   panics on `wasm32-unknown-unknown` (see [`NPS_ESTIMATE`]).

mod game;

pub use game::{Game, LimitsIn, SearchOut, StateOut};

use serde::Serialize;
use wasm_bindgen::prelude::*;

fn to_js<T: Serialize>(v: &T) -> Result<JsValue, JsError> {
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsError::new(&e.to_string()))
}

#[wasm_bindgen]
pub struct Engine {
    game: Game,
}

#[wasm_bindgen]
impl Engine {
    /// `generationJson` is a generation file (`gen-XXX.json`) or a bare constants object.
    #[wasm_bindgen(constructor)]
    pub fn new(generation_json: &str) -> Result<Engine, JsError> {
        Ok(Engine {
            game: Game::new(generation_json).map_err(|e| JsError::new(&e))?,
        })
    }

    /// Reset to the start position, or to `fen` (accreted mass starts at zero).
    #[wasm_bindgen(js_name = newGame)]
    pub fn new_game(&mut self, fen: Option<String>) -> Result<(), JsError> {
        self.game
            .new_game(fen.as_deref())
            .map_err(|e| JsError::new(&e))
    }

    /// Play a UCI move. Returns false if it is illegal or the game is over.
    pub fn play(&mut self, uci: &str) -> bool {
        self.game.play(uci)
    }

    #[wasm_bindgen(js_name = legalMoves)]
    pub fn legal_moves(&self) -> Vec<String> {
        self.game.legal_moves()
    }

    /// `{ fen, turn, result, reason, inCheck, ply, halfmoveClock }`
    pub fn state(&self) -> Result<JsValue, JsError> {
        to_js(&self.game.state())
    }

    /// The `02_LAWS_SPEC.md` §10 breakdown as a JS object (see `eval::Report`).
    pub fn breakdown(&self) -> Result<JsValue, JsError> {
        to_js(&self.game.breakdown_json())
    }

    /// `limits`: `{ depth?, movetimeMs?, nodes?, nps? }` (all optional).
    pub fn search(&mut self, limits: JsValue) -> Result<JsValue, JsError> {
        let l: LimitsIn = if limits.is_undefined() || limits.is_null() {
            LimitsIn::default()
        } else {
            serde_wasm_bindgen::from_value(limits).map_err(|e| JsError::new(&e.to_string()))?
        };
        let r = self.game.search(&l).map_err(|e| JsError::new(&e))?;
        to_js(&r)
    }

    /// Merge a partial constants object (`{"rho": 2.0}`) into the current constants.
    #[wasm_bindgen(js_name = setConstants)]
    pub fn set_constants(&mut self, partial_json: &str) -> Result<(), JsError> {
        self.game
            .set_constants(partial_json)
            .map_err(|e| JsError::new(&e))
    }

    /// The current constants as a JSON string.
    #[wasm_bindgen(js_name = constantsJson)]
    pub fn constants_json(&self) -> String {
        self.game.constants_json()
    }
}
