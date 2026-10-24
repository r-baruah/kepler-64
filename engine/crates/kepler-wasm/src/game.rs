//! Game state and the engine calls, free of any JS types.

use cozy_chess::{Board, Color, GameStatus, Piece};
use kepler_core::eval::{self, Report};
use kepler_core::params::{Generation, Params};
use kepler_core::position::Position;
use kepler_core::search::{Limits, Searcher, move_from_uci, move_to_uci};
use serde::{Deserialize, Serialize};

/// Nodes per second assumed when turning `movetimeMs` into a node budget.
///
/// Native kepler-core does roughly 0.5-1M nps on one thread; wasm in a desktop
/// browser is typically 1.5-2.5x slower, a phone 3-6x slower. 150k is a
/// conservative middle: a 1000 ms request searches at most 150k nodes, which
/// finishes in about a second on a desktop and a few seconds on a slow phone.
/// Callers who measured their device pass `nps` in the limits to override it.
pub const NPS_ESTIMATE: u64 = 150_000;

/// Depth used when `search` is called with no limits at all.
const DEFAULT_DEPTH: u32 = 4;

/// Transposition table size in the browser, MB.
const HASH_MB: usize = 16;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct LimitsIn {
    pub depth: Option<u32>,
    pub movetime_ms: Option<u64>,
    pub nodes: Option<u64>,
    /// Override of [`NPS_ESTIMATE`] for the movetime conversion.
    pub nps: Option<u64>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StateOut {
    pub fen: String,
    pub turn: &'static str,
    pub result: Option<&'static str>,
    pub reason: Option<&'static str>,
    pub in_check: bool,
    /// Half-moves played since `newGame`.
    pub ply: u32,
    pub halfmove_clock: u32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchOut {
    pub best: String,
    /// Score for the side to move, in Kepler units.
    pub score_ku: f32,
    pub mate_in: Option<i32>,
    pub depth: u32,
    pub seldepth: u32,
    pub nodes: u64,
    pub pv: Vec<String>,
}

/// Draws that cozy-chess does not detect by itself.
fn insufficient_material(b: &Board) -> bool {
    if !(b.pieces(Piece::Pawn) | b.pieces(Piece::Rook) | b.pieces(Piece::Queen)).is_empty() {
        return false;
    }
    let knights = b.pieces(Piece::Knight).len();
    let bishops = b.pieces(Piece::Bishop);
    if knights + bishops.len() <= 1 {
        return true;
    }
    if knights == 0 {
        let (mut light, mut dark) = (false, false);
        for sq in bishops {
            if (sq.file() as u8 + sq.rank() as u8) % 2 == 1 {
                light = true;
            } else {
                dark = true;
            }
        }
        return !(light && dark);
    }
    false
}

pub struct Game {
    params: Params,
    generation: Option<u32>,
    pos: Position,
    /// Hashes of every position since `newGame`, the current one last.
    hashes: Vec<u64>,
    searcher: Searcher,
}

impl Game {
    pub fn new(generation_json: &str) -> Result<Game, String> {
        let (params, generation) = parse_constants(generation_json)?;
        let pos = Position::startpos();
        Ok(Game {
            params,
            generation,
            hashes: vec![pos.board.hash()],
            pos,
            searcher: Searcher::new(HASH_MB),
        })
    }

    pub fn new_game(&mut self, fen: Option<&str>) -> Result<(), String> {
        let pos = match fen {
            None => Position::startpos(),
            Some(f) => {
                let f = f.trim();
                let f = if f.split_whitespace().count() == 4 {
                    format!("{f} 0 1")
                } else {
                    f.to_string()
                };
                Position::from_fen(&f)?
            }
        };
        self.hashes = vec![pos.board.hash()];
        self.pos = pos;
        self.searcher.clear();
        Ok(())
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn position(&self) -> &Position {
        &self.pos
    }

    pub fn legal_moves(&self) -> Vec<String> {
        let b = &self.pos.board;
        let mut v = Vec::with_capacity(40);
        b.generate_moves(|pm| {
            v.extend(pm.into_iter().map(|m| move_to_uci(b, m)));
            false
        });
        v
    }

    fn has_moves(&self) -> bool {
        self.pos.board.generate_moves(|_| true)
    }

    /// Game result and reason, if the game is over.
    pub fn result(&self) -> Option<(&'static str, &'static str)> {
        let b = &self.pos.board;
        match b.status() {
            GameStatus::Won => {
                // The side to move is mated.
                let r = if b.side_to_move() == Color::White {
                    "0-1"
                } else {
                    "1-0"
                };
                return Some((r, "checkmate"));
            }
            GameStatus::Drawn => {
                let why = if self.has_moves() {
                    "fifty_move_rule"
                } else {
                    "stalemate"
                };
                return Some(("1/2-1/2", why));
            }
            GameStatus::Ongoing => {}
        }
        if insufficient_material(b) {
            return Some(("1/2-1/2", "insufficient_material"));
        }
        let cur = *self.hashes.last().expect("history is never empty");
        if self.hashes.iter().filter(|&&h| h == cur).count() >= 3 {
            return Some(("1/2-1/2", "threefold_repetition"));
        }
        None
    }

    pub fn state(&self) -> StateOut {
        let b = &self.pos.board;
        let r = self.result();
        StateOut {
            fen: b.to_string(),
            turn: if b.side_to_move() == Color::White {
                "w"
            } else {
                "b"
            },
            result: r.map(|x| x.0),
            reason: r.map(|x| x.1),
            in_check: !b.checkers().is_empty(),
            ply: self.hashes.len() as u32 - 1,
            halfmove_clock: b.halfmove_clock() as u32,
        }
    }

    pub fn play(&mut self, uci: &str) -> bool {
        if self.result().is_some() {
            return false;
        }
        let Some(mv) = move_from_uci(&self.pos.board, uci) else {
            return false;
        };
        self.pos = self.pos.play(mv, &self.params);
        self.hashes.push(self.pos.board.hash());
        true
    }

    pub fn breakdown(&self) -> Report {
        eval::report(&self.pos, &self.params, self.generation)
    }

    /// The breakdown as JSON for the browser: `constants.lambda_tide` is always present
    /// (the file formats omit it at 0; the website should not have to guess).
    pub fn breakdown_json(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self.breakdown()).expect("Report serialises");
        v["constants"] = constants_value(&self.params);
        v
    }

    pub fn search(&mut self, l: &LimitsIn) -> Result<SearchOut, String> {
        if self.result().is_some() || !self.has_moves() {
            return Err("the game is over: no move to search".into());
        }
        let nps = l.nps.unwrap_or(NPS_ESTIMATE).max(1);
        let from_time = l
            .movetime_ms
            .map(|ms| (ms.saturating_mul(nps) / 1000).max(1));
        let nodes = match (l.nodes, from_time) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let depth = match (l.depth, nodes) {
            (None, None) => Some(DEFAULT_DEPTH),
            (d, _) => d,
        };
        let limits = Limits {
            depth,
            nodes,
            ..Default::default()
        };
        let history = &self.hashes[..self.hashes.len() - 1];
        let r = self
            .searcher
            .search(&self.pos, history, &self.params, &limits, &mut |_| {});
        let b = &self.pos.board;
        let best = r.best.ok_or("search returned no move")?;
        Ok(SearchOut {
            best: move_to_uci(b, best),
            score_ku: r.score_ku,
            mate_in: r.mate_in,
            depth: r.depth,
            seldepth: r.seldepth,
            nodes: r.nodes,
            pv: pv_to_uci(&self.pos, &r.pv),
        })
    }

    pub fn set_constants(&mut self, partial_json: &str) -> Result<(), String> {
        let partial: serde_json::Value =
            serde_json::from_str(partial_json).map_err(|e| e.to_string())?;
        let serde_json::Value::Object(patch) = partial else {
            return Err("constants must be a JSON object".into());
        };
        let mut cur = constants_value(&self.params);
        let obj = cur.as_object_mut().expect("Params serialises to an object");
        for (k, v) in patch {
            if !obj.contains_key(&k) {
                return Err(format!("unknown constant '{k}'"));
            }
            obj.insert(k, v);
        }
        let p: Params = serde_json::from_value(cur).map_err(|e| e.to_string())?;
        p.validate()?;
        self.params = p;
        // Scores in the table belong to the old universe.
        self.searcher.clear();
        Ok(())
    }

    pub fn constants_json(&self) -> String {
        constants_value(&self.params).to_string()
    }
}

/// The constants as a JSON object that always has `lambda_tide` (`Params` omits it at 0).
fn constants_value(p: &Params) -> serde_json::Value {
    let mut v = serde_json::to_value(p).expect("Params serialises");
    v["lambda_tide"] = serde_json::json!(p.lambda_tide);
    v
}

/// PV moves as UCI, replaying the line so castling is written correctly.
fn pv_to_uci(root: &Position, pv: &[cozy_chess::Move]) -> Vec<String> {
    let mut b = root.board.clone();
    let mut out = Vec::with_capacity(pv.len());
    for &m in pv {
        if !b.is_legal(m) {
            break;
        }
        out.push(move_to_uci(&b, m));
        b.play_unchecked(m);
    }
    out
}

fn parse_constants(json: &str) -> Result<(Params, Option<u32>), String> {
    if json.trim().is_empty() {
        return Ok((Params::default(), None));
    }
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    if v.get("constants").is_some() {
        let g = Generation::from_json(json)?;
        Ok((g.constants, Some(g.generation)))
    } else {
        let p: Params = serde_json::from_value(v).map_err(|e| e.to_string())?;
        p.validate()?;
        Ok((p, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEN2: &str = include_str!("../../../data/generations/gen-002.json");

    fn game() -> Game {
        Game::new(GEN2).unwrap()
    }

    #[test]
    fn plays_legal_moves_and_rejects_illegal() {
        let mut g = game();
        assert_eq!(g.legal_moves().len(), 20);
        assert!(!g.play("e2e5"));
        assert!(!g.play("garbage"));
        assert!(g.play("e2e4"));
        assert_eq!(g.state().turn, "b");
        assert_eq!(g.state().ply, 1);
        assert!(
            g.state()
                .fen
                .starts_with("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")
        );
        assert_eq!(g.breakdown().generation, Some(2));
    }

    #[test]
    fn castling_uses_standard_uci() {
        let mut g = game();
        g.new_game(Some("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
            .unwrap();
        let moves = g.legal_moves();
        assert!(moves.contains(&"e1g1".to_string()) && moves.contains(&"e1c1".to_string()));
        assert!(!moves.contains(&"e1h1".to_string()));
        assert!(g.play("e1g1"));
        assert!(g.state().fen.starts_with("r3k2r/8/8/8/8/8/8/R4RK1 b"));
    }

    #[test]
    fn lambda_tide_is_settable_and_reported() {
        let mut g = game();
        let c: serde_json::Value = serde_json::from_str(&g.constants_json()).unwrap();
        assert_eq!(c["lambda_tide"], 0.0);
        assert!(g.set_constants("{\"lambda_tide\": 0.34}").is_ok());
        assert!(g.set_constants("{\"lambda_tide\": 3.0}").is_err());
        let b = g.breakdown_json();
        assert!((b["constants"]["lambda_tide"].as_f64().unwrap() - 0.34).abs() < 1e-6);
        assert!(b["terms"]["tide"].is_number());
        let t = &b["terms"];
        let sum = t["material"].as_f64().unwrap()
            + t["bind"].as_f64().unwrap()
            + t["space"].as_f64().unwrap()
            + t["roche"].as_f64().unwrap()
            + t["tide"].as_f64().unwrap();
        assert!((sum - t["score_white"].as_f64().unwrap()).abs() < 1e-4);
        // A spec-1 generation file with lambda_tide > 0 is refused.
        assert!(
            Game::new(
                "{\"spec\": \"v1.0-spec-1\", \"generation\": 1, \"constants\": {\"lambda_tide\": 0.1}}"
            )
            .is_err()
        );
    }

    #[test]
    fn accretion_is_carried_through_play() {
        let mut g = Game::new("{\"f_acc\": 0.8}").unwrap();
        for m in ["e2e4", "d7d5", "e4d5"] {
            assert!(g.play(m));
        }
        let r = g.breakdown();
        let pawn = r.pieces.iter().find(|p| p.square == "d5").unwrap();
        assert_eq!(pawn.piece, 'P');
        assert!((pawn.accreted - 0.8).abs() < 1e-6);
        assert!((pawn.carried_mass - 1.8).abs() < 1e-6);
    }

    #[test]
    fn detects_checkmate() {
        let mut g = game();
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            assert!(g.play(m));
        }
        let s = g.state();
        assert_eq!(s.result, Some("0-1"));
        assert_eq!(s.reason, Some("checkmate"));
        assert!(!g.play("a2a3"));
        assert!(g.search(&LimitsIn::default()).is_err());
    }

    #[test]
    fn detects_stalemate_fifty_and_insufficient() {
        let mut g = game();
        g.new_game(Some("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")).unwrap();
        assert_eq!(g.state().reason, Some("stalemate"));
        g.new_game(Some("4k3/8/8/8/8/8/8/3RK3 w - - 100 80"))
            .unwrap();
        assert_eq!(g.state().reason, Some("fifty_move_rule"));
        g.new_game(Some("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")).unwrap();
        assert_eq!(g.state().reason, Some("insufficient_material"));
        g.new_game(Some("4k3/8/8/8/8/8/8/3RK3 w - - 0 1")).unwrap();
        assert_eq!(g.state().result, None);
    }

    #[test]
    fn detects_threefold_repetition() {
        let mut g = game();
        for m in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"] {
            assert!(g.play(m));
            assert_eq!(g.state().result, None);
        }
        assert!(g.play("f6g8"));
        let s = g.state();
        assert_eq!(s.result, Some("1/2-1/2"));
        assert_eq!(s.reason, Some("threefold_repetition"));
    }

    #[test]
    fn search_finds_mate_in_one_and_respects_limits() {
        let mut g = game();
        g.new_game(Some("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"))
            .unwrap();
        let r = g
            .search(&LimitsIn {
                depth: Some(3),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(r.best, "a1a8");
        assert_eq!(r.mate_in, Some(1));
        g.new_game(None).unwrap();
        let r = g
            .search(&LimitsIn {
                movetime_ms: Some(20),
                nps: Some(100_000),
                ..Default::default()
            })
            .unwrap();
        // 20 ms at 100k nps = 2000 nodes; the search may overshoot by one abort check.
        assert!(r.nodes <= 2500, "nodes {}", r.nodes);
        assert!(r.depth >= 1);
        assert!(g.legal_moves().contains(&r.best));
        let r = g
            .search(&LimitsIn {
                depth: Some(2),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(r.depth, 2);
    }

    #[test]
    fn set_constants_merges_and_validates() {
        let mut g = game();
        let rho0 = g.params().rho;
        let m_q0 = g.params().m_q;
        g.set_constants("{\"rho\": 2.5}").unwrap();
        assert_eq!(g.params().rho, 2.5);
        assert_eq!(g.params().m_q, m_q0);
        assert_ne!(rho0, 2.5);
        assert!(g.set_constants("{\"rho\": 99}").is_err());
        assert!(g.set_constants("{\"nope\": 1}").is_err());
        assert_eq!(g.params().rho, 2.5);
        assert_eq!(g.breakdown().constants.rho, 2.5);
    }

    #[test]
    fn bare_constants_and_bad_generation() {
        assert!(Game::new("{\"m_q\": 9.5}").is_ok());
        assert!(Game::new("{\"spec\": \"v0\", \"generation\": 1, \"constants\": {}}").is_err());
        assert!(Game::new("not json").is_err());
    }
}
