//! `kepler-train tdleaf`: TDLeaf(lambda) rounds (`docs/v1/04_TRAINING_AND_GATES.md` §1).
//!
//! Conventions
//! * `v_t = tanh(eval_white(leaf_t) / T)` in the view of the side that moved at ply `t`;
//!   a mate score gives `v_t = +-1` with zero gradient. `leaf_t` is the quiet leaf (T5):
//!   the end of the quiescence line from the PV leaf. `T` is the learned temperature (T1).
//! * `d_t = -v_{t+1} - v_t` (next value negated into the same view); the last ply uses the
//!   game result for its mover (+1 / 0 / -1). Games stopped by the max-plies cap follow
//!   `--capped-target` (T6): `drop` (default) gives them no gradient at all, `score` uses
//!   `tanh(final root search score / T)` as the last target, `draw` is the old behaviour.
//! * Update direction `sum_t grad v_t * e_t`, `e_t = d_t - lambda * e_{t+1}`. The spec's
//!   `sum_{j>=t} lambda^{j-t} d_j` is written with every `d_j` in one common view; with
//!   `d_j` in its own mover's view, ply `j`'s error enters ply `t`'s sum with sign
//!   `(-1)^{j-t}`, which is what the recursion implements. (With lambda = 0 the two agree.)
//! * Gradient ASCENT on agreement: `v` moves toward its target, so theta += step.
//! * Adam runs on theta normalised by its bound range `hi - lo`, so `--lr 0.01` means
//!   "about 1% of a constant's allowed range per round" for all 12 constants alike. The
//!   result is projected onto the bounds.
//! * `c` and `f_acc` are left unchanged (outer-loop constants).
//! * Temperature (T1): `T` is parameterised as `log T` (bounds `T` in [0.5, 50]) and
//!   learned by the same Adam (same relative learning rate, relative to the `log T` bound
//!   range) from `dv/dlogT = T * dv/dT = -(1 - v^2) * eval / T`, summed with the same `e_t`
//!   weights as the constants (`grad_logT_sum`). `--no-learn-temperature` keeps it fixed.
//!   `T` is not a law constant: it lives in the checkpoint and `training.temperature`.
//! * `--conserve-mass` (T2): after each step the four masses are rescaled by one common
//!   factor so `8 + 2 m_n + 2 m_b + 2 m_r + m_q` equals its value at the start of the run.
//!
//! Work units
//! * A round is "N shards, then reduce". `play_shard` plays games and returns the
//!   un-normalised sums (`Shard`); `reduce_round` validates shards, sums them, and takes one
//!   Adam step. `tdleaf` runs both in-process; `tdleaf-shard` / `tdleaf-reduce` are the same
//!   two functions behind separate commands (so shards can run on other machines).
//! * Shard seed of shard `k` in a round with round seed `R`: `util::mix(R, k)`. Inside a
//!   shard with seed `S`: first opening index `XorShift::new(S).below(n_openings)`, game `i`
//!   uses opening `(first + i/2) % n` (both colours) and game seed `util::mix(S, i)`.
//! * `gen_hash` ties shards to the constants they were played with (`selfgame::gen_hash`).

#![allow(clippy::needless_range_loop)]

use crate::chess::End;
use crate::selfgame::{
    self, AdjArgs, GameCfg, Leaf, constants_json, gen_hash, load_generation, load_openings, par_run,
};
use crate::util::{self, XorShift};
use clap::{Args, ValueEnum};
use kepler_core::params::{Generation, is_known_spec, spec_id};
use kepler_core::search::{Limits, Searcher};
use kepler_core::{NGRAD, Params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Search and game options shared by `tdleaf` and `tdleaf-shard`.
#[derive(Args, Debug, Clone)]
pub struct PlayOpts {
    #[arg(long, default_value_t = 4)]
    depth: u32,
    /// Milliseconds per move instead of --depth. Refused unless --allow-nondeterministic
    /// (rule D3); the shard then says "deterministic": false
    #[arg(long)]
    movetime: Option<u64>,
    #[arg(long = "allow-nondeterministic")]
    allow_nondeterministic: bool,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long, default_value_t = 0.7)]
    lambda: f32,
    /// Temperature T (ku) in tanh(eval / T), bounds [0.5, 50]. Default: the generation
    /// file's `training.temperature` if present, else 4.0 (a checkpoint overrides it)
    #[arg(long)]
    temperature: Option<f32>,
    /// What a game stopped by --max-plies teaches: drop (no gradient), score
    /// (tanh(final root score / T) as the final target), draw (the result 0)
    #[arg(long = "capped-target", value_enum, default_value_t = CappedTarget::Drop)]
    capped_target: CappedTarget,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 8)]
    explore_plies: usize,
    /// Transposition table per searcher, MB
    #[arg(long, default_value_t = 8)]
    hash: usize,
    /// spec-2: also report the gradient of `lambda_tide` (a 13th grad_sum entry) so it can
    /// be learned from 0. Off by default: a shard of a `lambda_tide = 0` universe is then
    /// exactly a spec-1 shard (invariant 9). Always on when `lambda_tide > 0`.
    #[arg(long = "learn-tide")]
    learn_tide: bool,
    #[command(flatten)]
    adj: AdjArgs,
}

/// T6: how games stopped at the ply cap enter the TD targets.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CappedTarget {
    Drop,
    Score,
    Draw,
}

impl CappedTarget {
    pub fn name(self) -> &'static str {
        match self {
            CappedTarget::Drop => "drop",
            CappedTarget::Score => "score",
            CappedTarget::Draw => "draw",
        }
    }
}

/// Temperature bounds (ku) and its Adam range on the log scale (T1).
pub const T_BOUNDS: (f32, f32) = (0.5, 50.0);
pub const DEFAULT_T: f32 = 4.0;

fn logt_range() -> f64 {
    libm::log(T_BOUNDS.1 as f64) - libm::log(T_BOUNDS.0 as f64)
}

/// Initial temperature: the explicit flag, else the generation's `training.temperature`,
/// else 4.0.
pub fn initial_temperature(explicit: Option<f32>, gen_: &Generation) -> Result<f32, String> {
    let t = explicit
        .or_else(|| {
            gen_.training
                .get("temperature")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
        })
        .unwrap_or(DEFAULT_T);
    if !(T_BOUNDS.0..=T_BOUNDS.1).contains(&t) {
        return Err(format!(
            "temperature {t} is outside [{}, {}]",
            T_BOUNDS.0, T_BOUNDS.1
        ));
    }
    Ok(t)
}

/// Update options shared by `tdleaf` and `tdleaf-reduce`.
#[derive(Args, Debug, Clone)]
pub struct UpdateOpts {
    /// Learn the temperature T alongside the constants (the default)
    #[arg(long = "learn-temperature", num_args = 0..=1, default_missing_value = "true", default_value_t = true)]
    learn_temperature: bool,
    /// Keep the temperature fixed (same as --learn-temperature false)
    #[arg(long = "no-learn-temperature")]
    no_learn_temperature: bool,
    /// After every step rescale the four masses so 8 + 2 m_n + 2 m_b + 2 m_r + m_q stays
    /// at its starting value
    #[arg(long = "conserve-mass")]
    conserve_mass: bool,
}

impl UpdateOpts {
    fn learn_t(&self) -> bool {
        self.learn_temperature && !self.no_learn_temperature
    }
}

#[derive(Args, Debug)]
pub struct TdleafArgs {
    /// Starting generation (parent of the candidate)
    #[arg(long = "gen")]
    gen_file: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long = "out-dir")]
    out_dir: PathBuf,
    /// Target total number of rounds (with --resume, counting the rounds already done)
    #[arg(long, default_value_t = 1)]
    rounds: usize,
    /// Games per round in total, split evenly over --shards
    #[arg(long = "games-per-round", default_value_t = 2000)]
    games_per_round: usize,
    /// In-process shards per round (played one after another, each with all threads)
    #[arg(long, default_value_t = 1)]
    shards: usize,
    /// Adam learning rate in units of each constant's bound range
    #[arg(long, default_value_t = 0.01)]
    lr: f32,
    #[arg(long = "max-minutes")]
    max_minutes: Option<f64>,
    #[arg(long)]
    resume: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[command(flatten)]
    play: PlayOpts,
    #[command(flatten)]
    upd: UpdateOpts,
}

#[derive(Args, Debug)]
pub struct ShardArgs {
    /// Generation file whose constants play (both sides)
    #[arg(long = "gen")]
    gen_file: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long, default_value_t = 100)]
    games: usize,
    /// The shard seed; with --shard-index it is the round seed and the shard seed is
    /// mix(seed, shard-index)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long = "shard-index")]
    shard_index: Option<u64>,
    #[command(flatten)]
    play: PlayOpts,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args, Debug)]
pub struct ReduceArgs {
    /// Previous checkpoint-NNN.json, or "none" to start from --gen
    #[arg(long, default_value = "none")]
    state: String,
    /// The generation the shards were played with (a parent gen, or the previous
    /// candidate.json); its constants must match the shards' gen_hash
    #[arg(long = "gen")]
    gen_file: PathBuf,
    /// Shard files, summed in the order given
    #[arg(long, num_args = 1.., conflicts_with = "shard_dir")]
    shards: Vec<PathBuf>,
    /// Directory of shard *.json files (summed in file-name order)
    #[arg(long = "shard-dir")]
    shard_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0.01)]
    lr: f32,
    /// Initial temperature when --state is "none" (default: --gen's
    /// training.temperature, else 4.0)
    #[arg(long)]
    temperature: Option<f32>,
    #[command(flatten)]
    upd: UpdateOpts,
    #[arg(long = "out-dir")]
    out_dir: PathBuf,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShardResults {
    pub w: usize,
    pub d: usize,
    pub l: usize,
    pub capped: usize,
    pub adjudicated: usize,
}

/// One work unit of a round (JSON file written by `tdleaf-shard`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shard {
    pub kind: String,
    pub version: u32,
    pub spec: String,
    pub gen_hash: String,
    pub seed: u64,
    pub games: usize,
    pub plies: usize,
    /// Un-normalised sum over all plies of grad(v_t) * e_t.
    pub grad_sum: Vec<f64>,
    /// Un-normalised sum over all plies of dv/dlogT * e_t (T = temperature).
    #[serde(rename = "grad_logT_sum", default)]
    pub grad_logt_sum: f64,
    /// Plies of capped games that contributed no gradient (`--capped-target drop`).
    #[serde(default)]
    pub dropped_plies: usize,
    pub abs_d_sum: f64,
    pub d_sum: f64,
    pub results: ShardResults,
    pub elapsed_ms: u64,
    pub config: serde_json::Value,
    /// False only for units run with `--allow-nondeterministic` (rule D3).
    #[serde(default = "util::yes")]
    pub deterministic: bool,
    /// Canonical hash of the deterministic content (rule D6, `util::canonical_hash`).
    #[serde(default)]
    pub result_hash: Option<String>,
}

pub fn shard_seed(round_seed: u64, k: u64) -> u64 {
    util::mix(round_seed, k)
}

/// Play `games` self-play games of `params` against itself and sum the TD statistics.
pub fn play_shard(
    params: &Params,
    openings: &[selfgame::Opening],
    games: usize,
    seed: u64,
    o: &PlayOpts,
    temperature: f32,
) -> Result<Shard, String> {
    util::check_time_limit(o.movetime.is_some(), o.allow_nondeterministic)?;
    let cfg = GameCfg {
        limits: match o.movetime {
            Some(ms) => Limits {
                movetime_ms: Some(ms),
                ..Default::default()
            },
            None => Limits {
                depth: Some(o.depth),
                ..Default::default()
            },
        },
        explore_depth: o.depth.saturating_sub(1).max(1),
        max_plies: o.max_plies,
        explore_plies: o.explore_plies,
        explore_margin: 0.15,
        record: true,
        quiet_leaf: true,
        temperature,
        adj: o.adj.cfg(),
    };
    let first = XorShift::new(seed).below(openings.len());
    let t0 = Instant::now();
    let played = par_run(
        games,
        o.threads,
        || [Searcher::new(o.hash), Searcher::new(o.hash)],
        |st, i| {
            let op = &openings[(first + i / 2) % openings.len()];
            st[0].clear();
            st[1].clear();
            let [s0, s1] = st;
            let g = selfgame::play_game(
                &cfg,
                op,
                [s0, s1],
                [params, params],
                util::mix(seed, i as u64),
            )?;
            if g.moves.is_empty() {
                return Ok(None);
            }
            let last = g.moves.len() - 1;
            let result_last = match (g.capped, o.capped_target) {
                (false, _) | (true, CappedTarget::Draw) => Some(g.result_for_mover(last)),
                (true, CappedTarget::Score) => {
                    Some(libm::tanhf(g.last_score_ku.unwrap_or(0.0) / temperature))
                }
                (true, CappedTarget::Drop) => None,
            };
            let dropped = if result_last.is_none() {
                g.leaves.len()
            } else {
                0
            };
            let td = result_last.map(|r| game_td(&g.leaves, r, o.lambda, temperature));
            Ok::<_, String>(Some((td, dropped, g.result, g.capped, g.adjudicated)))
        },
    );
    let mut grad = vec![0.0f64; NGRAD];
    let mut sh = Shard {
        kind: "tdleaf-shard".into(),
        version: 1,
        spec: spec_id(params).into(),
        gen_hash: gen_hash(params),
        seed,
        games: 0,
        plies: 0,
        grad_sum: Vec::new(),
        grad_logt_sum: 0.0,
        dropped_plies: 0,
        abs_d_sum: 0.0,
        d_sum: 0.0,
        results: ShardResults::default(),
        elapsed_ms: 0,
        config: serde_json::Value::Null,
        deterministic: !o.allow_nondeterministic,
        result_hash: None,
    };
    for g in played {
        let Some((td, dropped, res, cap, adj)) = g? else {
            continue;
        };
        sh.games += 1;
        match res {
            End::WhiteWins => sh.results.w += 1,
            End::Draw => sh.results.d += 1,
            End::BlackWins => sh.results.l += 1,
        }
        sh.results.capped += cap as usize;
        sh.results.adjudicated += adj as usize;
        sh.dropped_plies += dropped;
        let Some(td) = td else {
            continue;
        };
        sh.plies += td.plies;
        sh.abs_d_sum += td.sum_abs_d;
        sh.d_sum += td.sum_d;
        sh.grad_logt_sum += td.grad_logt_sum;
        for i in 0..NGRAD {
            grad[i] += td.grad_sum[i] as f64;
        }
    }
    let tide_on = o.learn_tide || params.lambda_tide != 0.0;
    if !tide_on {
        grad.truncate(OLD_NGRAD);
    }
    sh.grad_sum = grad;
    sh.elapsed_ms = t0.elapsed().as_millis() as u64;
    sh.config = json!({
        "depth": o.depth, "movetime_ms": o.movetime,
        "threads": o.threads, "hash_mb": o.hash,
        "lambda": o.lambda, "temperature": temperature,
        "capped_target": o.capped_target.name(), "quiet_leaf": true,
        "max_plies": o.max_plies, "explore_plies": o.explore_plies, "explore_margin": 0.15,
        "games_requested": games,
        "adjudication": o.adj.cfg(),
        "created": util::utc_iso(),
    });
    if tide_on {
        sh.config["learn_tide"] = json!(true);
    }
    Ok(sh)
}

/// TD errors and the lambda-return for one game.
///
/// `vs[t]`: value at ply `t` in its mover's view. `result_last`: result for the last
/// mover. Returns `(d, e)` with `d_t` as above and `e_t = d_t - lambda * e_{t+1}`.
pub fn td_errors(vs: &[f32], result_last: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    let n = vs.len();
    let mut d = vec![0.0; n];
    for t in 0..n {
        let target = if t + 1 < n { -vs[t + 1] } else { result_last };
        d[t] = target - vs[t];
    }
    let mut e = vec![0.0; n];
    let mut next = 0.0;
    for t in (0..n).rev() {
        next = d[t] - lambda * next;
        e[t] = next;
    }
    (d, e)
}

/// Per-game contribution to a round.
pub struct GameTd {
    pub grad_sum: [f32; NGRAD],
    /// sum_t dv_t/dlogT * e_t
    pub grad_logt_sum: f64,
    pub plies: usize,
    pub sum_abs_d: f64,
    pub sum_d: f64,
}

pub fn game_td(leaves: &[Leaf], result_last: f32, lambda: f32, temperature: f32) -> GameTd {
    let vs: Vec<f32> = leaves.iter().map(|l| l.v).collect();
    let (d, e) = td_errors(&vs, result_last, lambda);
    let mut grad_sum = [0.0f32; NGRAD];
    let mut grad_logt_sum = 0.0f64;
    for (l, et) in leaves.iter().zip(&e) {
        for i in 0..NGRAD {
            grad_sum[i] += l.grad[i] * et;
        }
        // dv/dlogT = T dv/dT
        grad_logt_sum += (l.grad_t * temperature * et) as f64;
    }
    GameTd {
        grad_sum,
        grad_logt_sum,
        plies: leaves.len(),
        sum_abs_d: d.iter().map(|x| x.abs() as f64).sum(),
        sum_d: d.iter().map(|&x| x as f64).sum(),
    }
}

/// Number of gradient slots before spec-2 added `lambda_tide`.
const OLD_NGRAD: usize = 12;

/// Read an Adam moment vector; a 12-entry one from before spec-2 gets a zero moment for
/// `lambda_tide`.
fn de_moments<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[f64; NGRAD], D::Error> {
    let v = Vec::<f64>::deserialize(d)?;
    if v.len() != NGRAD && v.len() != OLD_NGRAD {
        return Err(serde::de::Error::custom(format!(
            "Adam moments have {} entries, expected {NGRAD} (or {OLD_NGRAD} from before spec-2)",
            v.len()
        )));
    }
    let mut out = [0.0; NGRAD];
    out[..v.len()].copy_from_slice(&v);
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adam {
    #[serde(deserialize_with = "de_moments")]
    pub m: [f64; NGRAD],
    #[serde(deserialize_with = "de_moments")]
    pub v: [f64; NGRAD],
    pub t: u64,
}

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;

impl Adam {
    pub fn new() -> Self {
        Adam {
            m: [0.0; NGRAD],
            v: [0.0; NGRAD],
            t: 0,
        }
    }

    /// One ascent step. `grad` is the ascent direction in raw constant units; Adam works
    /// on the range-normalised constant (`grad * range`) and the result is projected onto
    /// the bounds. Returns the new constants.
    pub fn step(&mut self, theta: [f32; NGRAD], grad: [f32; NGRAD], lr: f32) -> [f32; NGRAD] {
        self.t += 1;
        let (b1t, b2t) = (
            1.0 - libm::pow(BETA1, (self.t as i32) as f64),
            1.0 - libm::pow(BETA2, (self.t as i32) as f64),
        );
        let mut out = theta;
        for i in 0..NGRAD {
            let (lo, hi) = Params::GRAD_BOUNDS[i];
            let range = (hi - lo) as f64;
            let g = grad[i] as f64 * range;
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * g;
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * g * g;
            let step = lr as f64 * (self.m[i] / b1t) / ((self.v[i] / b2t).sqrt() + EPS);
            out[i] = (theta[i] as f64 + step * range) as f32;
        }
        out
    }
}

/// Adam for the single `log T` parameter (same hyper-parameters as `Adam`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdamT {
    pub m: f64,
    pub v: f64,
    pub t: u64,
}

impl AdamT {
    /// One ascent step on `T`. `grad` is the ascent direction in `log T` units; Adam works
    /// on the range-normalised `log T` (`grad * range`) with the relative rate `lr`, and
    /// the result is projected onto the bounds. Returns the new `T`.
    pub fn step(&mut self, temperature: f32, grad: f64, lr: f32) -> f32 {
        self.t += 1;
        let (b1t, b2t) = (
            1.0 - libm::pow(BETA1, (self.t as i32) as f64),
            1.0 - libm::pow(BETA2, (self.t as i32) as f64),
        );
        let range = logt_range();
        let g = grad * range;
        self.m = BETA1 * self.m + (1.0 - BETA1) * g;
        self.v = BETA2 * self.v + (1.0 - BETA2) * g * g;
        let step = lr as f64 * (self.m / b1t) / ((self.v / b2t).sqrt() + EPS);
        let lt = libm::log(temperature as f64) + step * range;
        libm::exp(lt.clamp(libm::log(T_BOUNDS.0 as f64), libm::log(T_BOUNDS.1 as f64)))
            .clamp(T_BOUNDS.0 as f64, T_BOUNDS.1 as f64) as f32
    }
}

/// `8 + 2 m_n + 2 m_b + 2 m_r + m_q`: the rest mass of a full army (eight pawns of mass 1).
pub fn total_mass(p: &Params) -> f64 {
    8.0 + 2.0 * (p.m_n + p.m_b + p.m_r) as f64 + p.m_q as f64
}

/// T2: rescale the four piece masses by one common factor so `total_mass == target`, then
/// clamp to the bounds (a clamp can leave the total off target).
pub fn conserve_mass(p: &mut Params, target: f64) {
    let have = total_mass(p) - 8.0;
    if have <= 0.0 || target <= 8.0 {
        return;
    }
    let f = ((target - 8.0) / have) as f32;
    let b = Params::GRAD_BOUNDS;
    p.m_n = (p.m_n * f).clamp(b[0].0, b[0].1);
    p.m_b = (p.m_b * f).clamp(b[1].0, b[1].1);
    p.m_r = (p.m_r * f).clamp(b[2].0, b[2].1);
    p.m_q = (p.m_q * f).clamp(b[3].0, b[3].1);
}

/// Constants sitting on a bound. `tide_learned`: whether `lambda_tide` is being trained
/// (13-entry gradients); while it isn't, its 0 is "unused", not "pinned".
pub fn at_bound(p: &Params, tide_learned: bool) -> Vec<String> {
    let mut v = Vec::new();
    for (i, x) in p.grad_values().iter().enumerate() {
        let (lo, hi) = Params::GRAD_BOUNDS[i];
        let tol = 1e-6 * (hi - lo);
        // spec-2: an unused lambda_tide (not trained, exactly 0) is not "pinned"; this keeps
        // the bound lists, and so the files that carry them, the same as under spec-1.
        // A trained lambda_tide that reaches 0 is reported like any other bound.
        if Params::GRAD_NAMES[i] == "lambda_tide" && !tide_learned && *x == 0.0 {
            continue;
        }
        if *x <= lo + tol || *x >= hi - tol {
            v.push(Params::GRAD_NAMES[i].to_string());
        }
    }
    v
}

/// Constants at a bound in more than 50% of the last 10 rounds.
pub fn pinned(history: &[Vec<String>]) -> Vec<String> {
    let last = &history[history.len().saturating_sub(10)..];
    Params::GRAD_NAMES
        .iter()
        .filter(|n| {
            let c = last.iter().filter(|r| r.iter().any(|x| x == *n)).count();
            2 * c > last.len()
        })
        .map(|s| s.to_string())
        .collect()
}

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub round: usize,
    pub games: usize,
    pub parent: u32,
    pub params: Params,
    pub adam: Adam,
    pub rng_state: u64,
    pub bound_history: Vec<Vec<String>>,
    /// Temperature T (ku), learned alongside the constants (not a law constant).
    #[serde(default = "default_t")]
    pub temperature: f32,
    #[serde(default)]
    pub adam_t: AdamT,
    /// `total_mass` of the run's starting generation (0 = not yet recorded).
    #[serde(default)]
    pub mass_total: f64,
}

fn default_t() -> f32 {
    DEFAULT_T
}

fn latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(usize, PathBuf)> = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(n) = name
            .strip_prefix("checkpoint-")
            .and_then(|s| s.strip_suffix(".json"))
            .and_then(|s| s.parse::<usize>().ok())
            && best.as_ref().is_none_or(|b| n > b.0)
        {
            best = Some((n, e.path()));
        }
    }
    best.map(|b| b.1)
}

#[derive(Debug)]
pub struct RoundSummary {
    pub round_json: serde_json::Value,
    pub candidate: Generation,
    pub text: String,
    pub finite: bool,
}

/// A numeric entry of a shard's `config` object.
fn num(config: &serde_json::Value, k: &str) -> Option<f64> {
    config.get(k)?.as_f64()
}

/// Options of one reduce step.
#[derive(Clone, Copy, Debug)]
pub struct StepOpts {
    pub lr: f32,
    pub learn_t: bool,
    pub conserve_mass: bool,
}

/// Validate `shards`, sum them, and take one Adam step on `ck` (which is advanced).
/// Shards must share spec and gen_hash, and that hash must be the hash of `ck.params`
/// (the constants the shards must have been played with).
pub fn reduce_round(
    ck: &mut Checkpoint,
    shards: &[Shard],
    step: StepOpts,
    spec: &str,
) -> Result<RoundSummary, String> {
    let lr = step.lr;
    // D7: combine in (shard_index, seed, result_hash) order, whatever order they arrive in
    let mut sorted = shards.to_vec();
    sorted.sort_by_key(|s| {
        (
            s.config
                .get("shard_index")
                .and_then(|x| x.as_u64())
                .unwrap_or(u64::MAX),
            s.seed,
            s.result_hash.clone(),
        )
    });
    let shards = &sorted[..];
    let first = shards.first().ok_or("no shards to reduce")?;
    util::check_same_determinism(shards.iter().map(|s| s.deterministic))?;
    let want = gen_hash(&ck.params);
    for (i, s) in shards.iter().enumerate() {
        if s.kind != "tdleaf-shard" || s.version != 1 {
            return Err(format!("shard {i}: not a tdleaf-shard v1 file"));
        }
        if s.spec != first.spec || s.spec != spec_id(&ck.params) || !is_known_spec(spec) {
            return Err(format!(
                "shard {i}: spec {} differs from {} / generation {spec} / constants {}",
                s.spec,
                first.spec,
                spec_id(&ck.params)
            ));
        }
        if s.gen_hash != first.gen_hash {
            return Err(format!(
                "shard {i}: gen_hash {} differs from shard 0 ({})",
                s.gen_hash, first.gen_hash
            ));
        }
        if s.gen_hash != want {
            return Err(format!(
                "shard {i}: gen_hash {} does not match the current constants ({want})",
                s.gen_hash
            ));
        }
        // A shard played before constant 13 existed has 12 entries (lambda_tide's partial is
        // then zero).
        if s.grad_sum.len() != first.grad_sum.len() {
            return Err(format!(
                "shard {i}: grad_sum has {} entries, shard 0 has {}",
                s.grad_sum.len(),
                first.grad_sum.len()
            ));
        }
        if s.grad_sum.len() != NGRAD && s.grad_sum.len() != OLD_NGRAD {
            return Err(format!(
                "shard {i}: grad_sum has {} entries",
                s.grad_sum.len()
            ));
        }
        for k in ["lambda", "temperature"] {
            if num(&s.config, k).is_none() || num(&s.config, k) != num(&first.config, k) {
                return Err(format!(
                    "shard {i}: config {k} is missing or differs from shard 0"
                ));
            }
        }
        if s.config.get("capped_target") != first.config.get("capped_target")
            || s.config.get("capped_target").is_none()
        {
            return Err(format!(
                "shard {i}: config capped_target is missing or differs from shard 0"
            ));
        }
        if num(&s.config, "temperature").map(|t| t as f32) != Some(ck.temperature) {
            return Err(format!(
                "shard {i}: temperature {:?} does not match the current temperature {}",
                num(&s.config, "temperature"),
                ck.temperature
            ));
        }
    }
    let mut grad = [0.0f64; NGRAD];
    let (mut grad_logt, mut dropped) = (0.0f64, 0usize);
    let (mut plies, mut games, mut sum_abs_d, mut sum_d, mut ms) = (0usize, 0usize, 0.0, 0.0, 0u64);
    let mut res = ShardResults::default();
    for s in shards {
        plies += s.plies;
        games += s.games;
        sum_abs_d += s.abs_d_sum;
        sum_d += s.d_sum;
        ms += s.elapsed_ms;
        grad_logt += s.grad_logt_sum;
        dropped += s.dropped_plies;
        res.w += s.results.w;
        res.d += s.results.d;
        res.l += s.results.l;
        res.capped += s.results.capped;
        res.adjudicated += s.results.adjudicated;
        for (g, x) in grad.iter_mut().zip(&s.grad_sum) {
            *g += x;
        }
    }
    let np = plies.max(1) as f64;
    let mut g32 = [0.0f32; NGRAD];
    for i in 0..NGRAD {
        g32[i] = (grad[i] / np) as f32;
    }
    let before = ck.params.grad_values();
    if ck.mass_total <= 0.0 {
        ck.mass_total = total_mass(&ck.params);
    }
    let proposed = ck.adam.step(before, g32, lr);
    let mut next = ck.params;
    next.set_grad_values(proposed);
    if step.conserve_mass {
        conserve_mass(&mut next, ck.mass_total);
    }
    let after = next.grad_values();
    let g_logt = grad_logt / np;
    let t_before = ck.temperature;
    if step.learn_t {
        ck.temperature = ck.adam_t.step(ck.temperature, g_logt, lr);
    }
    let delta: Vec<f32> = (0..NGRAD).map(|i| after[i] - before[i]).collect();
    ck.params = next;
    ck.round += 1;
    ck.games += games;
    let bound = at_bound(&ck.params, first.grad_sum.len() == NGRAD);
    ck.bound_history.push(bound.clone());
    let pin = pinned(&ck.bound_history);
    let finite = g32.iter().all(|x| x.is_finite())
        && after.iter().all(|x| x.is_finite())
        && g_logt.is_finite()
        && ck.temperature.is_finite();
    let secs = ms as f64 / 1000.0;
    let (depth, lambda) = (
        first.config.get("depth").cloned().unwrap_or_default(),
        first.config.get("lambda").cloned().unwrap_or_default(),
    );
    let capped_target = first
        .config
        .get("capped_target")
        .cloned()
        .unwrap_or_default();
    let mut names_delta = serde_json::Map::new();
    let mut names_grad = serde_json::Map::new();
    // A 12-entry shard set does not learn lambda_tide: its keys and lines are left out so
    // the round files read as under spec-1.
    let shown = first.grad_sum.len();
    for i in 0..shown {
        names_delta.insert(Params::GRAD_NAMES[i].into(), json!(delta[i]));
        names_grad.insert(Params::GRAD_NAMES[i].into(), json!(g32[i]));
    }
    let round_json = json!({
        "round": ck.round, "games": games, "plies": plies,
        "wdl": { "white_wins": res.w, "draws": res.d, "black_wins": res.l,
                 "capped": res.capped, "adjudicated": res.adjudicated },
        "mean_plies": plies as f64 / games.max(1) as f64,
        "mean_abs_d": sum_abs_d / np, "mean_d": sum_d / np,
        "games_per_sec": games as f64 / secs.max(1e-9), "shard_seconds_sum": secs,
        "shards": shards.iter().map(|s| json!({
            "seed": s.seed, "games": s.games, "plies": s.plies, "elapsed_ms": s.elapsed_ms,
        })).collect::<Vec<_>>(),
        "gen_hash": want,
        "grad_avg": names_grad, "delta": names_delta,
        "constants": constants_json(&ck.params),
        "at_bound": bound, "pinned": pin,
        "depth": depth, "lambda": lambda, "lr": lr,
        "temperature": { "before": t_before, "after": ck.temperature,
                         "grad_logT": g_logt, "learned": step.learn_t },
        "capped_target": capped_target, "dropped_plies": dropped,
        "conserve_mass": step.conserve_mass, "mass_total": ck.mass_total,
        "adjudication": first.config.get("adjudication").cloned().unwrap_or_default(),
        "finite": finite,
    });
    let candidate = Generation {
        generation: ck.parent + 1,
        parent: Some(ck.parent),
        // spec-1 while lambda_tide is still 0 (the file then omits it), spec-2 once it is learned
        spec: spec_id(&ck.params).to_string(),
        created: util::utc_date(),
        constants: ck.params,
        pinned: pin.clone(),
        training: json!({
            "rounds": ck.round, "games": ck.games, "depth": depth,
            "lambda": lambda, "lr": lr,
            "temperature": ck.temperature, "learn_temperature": step.learn_t,
            "capped_target": capped_target, "conserve_mass": step.conserve_mass,
            "lr_units": "relative to bound range (hi - lo)", "optimizer": "adam",
            "games_per_round": games, "shards_per_round": shards.len(),
        }),
        acceptance: serde_json::Value::Null,
    };
    let mut text = format!(
        "round {} | {} games from {} shards ({:.2}/s, {:.0}s of shard time) | W{} D{} L{} cap{} adj{} | mean plies {:.1} | mean|d| {:.4} mean d {:+.4} | dropped plies {} | finite {}\n",
        ck.round,
        games,
        shards.len(),
        games as f64 / secs.max(1e-9),
        secs,
        res.w,
        res.d,
        res.l,
        res.capped,
        res.adjudicated,
        plies as f64 / games.max(1) as f64,
        sum_abs_d / np,
        sum_d / np,
        dropped,
        finite
    );
    text += &format!(
        "  temperature   {:>9.5} -> {:>9.5}  grad_logT {:+.3e}{}\n",
        t_before,
        ck.temperature,
        g_logt,
        if step.learn_t { "" } else { " (fixed)" }
    );
    for i in 0..shown {
        text += &format!(
            "  {:<13} {:>9.5} -> {:>9.5} ({:+.5})  grad {:+.3e}\n",
            Params::GRAD_NAMES[i],
            before[i],
            after[i],
            delta[i],
            g32[i]
        );
    }
    if !pin.is_empty() {
        text += &format!("  pinned: {}\n", pin.join(", "));
    }
    Ok(RoundSummary {
        round_json,
        candidate,
        text,
        finite,
    })
}

pub(crate) fn write_json(path: PathBuf, v: &serde_json::Value) -> Result<(), String> {
    std::fs::write(&path, serde_json::to_string_pretty(v).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Write round-NNN.json, checkpoint-NNN.json and candidate.json.
fn write_round(dir: &Path, ck: &Checkpoint, s: &RoundSummary) -> Result<(), String> {
    write_json(
        dir.join(format!("round-{:03}.json", ck.round)),
        &s.round_json,
    )?;
    write_json(
        dir.join(format!("checkpoint-{:03}.json", ck.round)),
        &serde_json::to_value(ck).unwrap(),
    )?;
    write_json(
        dir.join("candidate.json"),
        &serde_json::to_value(&s.candidate).unwrap(),
    )
}

fn fresh_checkpoint(parent: &Generation, seed: u64, temperature: f32) -> Checkpoint {
    Checkpoint {
        round: 0,
        games: 0,
        parent: parent.generation,
        params: parent.constants,
        adam: Adam::new(),
        rng_state: XorShift::new(seed).state(),
        bound_history: Vec::new(),
        temperature,
        adam_t: AdamT::default(),
        mass_total: total_mass(&parent.constants),
    }
}

pub fn run(a: TdleafArgs) -> Result<(), String> {
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let parent = load_generation(&a.gen_file)?;
    let openings = load_openings(&a.openings)?;
    if a.shards == 0 || a.games_per_round < a.shards {
        return Err("--shards must be >= 1 and <= --games-per-round".into());
    }
    let t_init = initial_temperature(a.play.temperature, &parent)?;
    let mut ck = fresh_checkpoint(&parent, a.seed, t_init);
    if a.resume {
        let p = latest_checkpoint(&a.out_dir).ok_or("--resume: no checkpoint in out-dir")?;
        let s = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
        ck = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))?;
        eprintln!("resumed from {} (round {})", p.display(), ck.round);
    }
    let mut rng = XorShift::from_state(ck.rng_state);
    let t_start = Instant::now();
    while ck.round < a.rounds {
        let round_seed = rng.next_u64();
        let mut shards = Vec::new();
        for k in 0..a.shards {
            // even split, the first (games % shards) shards take one extra
            let n = a.games_per_round / a.shards + usize::from(k < a.games_per_round % a.shards);
            let mut sh = play_shard(
                &ck.params,
                &openings,
                n,
                shard_seed(round_seed, k as u64),
                &a.play,
                ck.temperature,
            )?;
            sh.config["round_seed"] = json!(round_seed);
            sh.config["shard_index"] = json!(k);
            shards.push(sh);
        }
        let step = StepOpts {
            lr: a.lr,
            learn_t: a.upd.learn_t(),
            conserve_mass: a.upd.conserve_mass,
        };
        let sum = reduce_round(&mut ck, &shards, step, &parent.spec)?;
        ck.rng_state = rng.state();
        write_round(&a.out_dir, &ck, &sum)?;
        print!("{}", sum.text);
        if !sum.finite {
            return Err("non-finite update; stopping".into());
        }
        if let Some(m) = a.max_minutes
            && t_start.elapsed().as_secs_f64() / 60.0 >= m
        {
            println!("--max-minutes reached after round {}", ck.round);
            break;
        }
    }
    Ok(())
}

pub fn run_shard(a: ShardArgs) -> Result<(), String> {
    let gen_ = load_generation(&a.gen_file)?;
    let openings = load_openings(&a.openings)?;
    let seed = match a.shard_index {
        Some(k) => shard_seed(a.seed, k),
        None => a.seed,
    };
    let temperature = initial_temperature(a.play.temperature, &gen_)?;
    let mut sh = play_shard(
        &gen_.constants,
        &openings,
        a.games,
        seed,
        &a.play,
        temperature,
    )?;
    sh.config["generation"] = json!(gen_.generation);
    sh.config["gen_file"] = json!(a.gen_file);
    sh.config["openings"] = json!(a.openings);
    sh.config["openings_hash"] = json!(util::file_hash(&a.openings)?);
    if let Some(k) = a.shard_index {
        sh.config["round_seed"] = json!(a.seed);
        sh.config["shard_index"] = json!(k);
    }
    sh.result_hash = Some(util::result_hash_of(&sh));
    write_json(a.out.clone(), &serde_json::to_value(&sh).unwrap())?;
    println!(
        "shard {}: {} games, {} plies in {:.1}s ({:.2} games/s) | W{} D{} L{} cap{} adj{} | gen_hash {} seed {}",
        a.out.display(),
        sh.games,
        sh.plies,
        sh.elapsed_ms as f64 / 1000.0,
        sh.games as f64 / (sh.elapsed_ms as f64 / 1000.0).max(1e-9),
        sh.results.w,
        sh.results.d,
        sh.results.l,
        sh.results.capped,
        sh.results.adjudicated,
        sh.gen_hash,
        sh.seed
    );
    Ok(())
}

fn read_shards(a: &ReduceArgs) -> Result<Vec<Shard>, String> {
    let mut files = a.shards.clone();
    if let Some(d) = &a.shard_dir {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d)
            .map_err(|e| format!("{}: {e}", d.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        v.sort();
        files = v;
    }
    if files.is_empty() {
        return Err("give --shards <files...> or --shard-dir".into());
    }
    files.iter().map(|p| util::load_shard(p)).collect()
}

pub fn run_reduce(a: ReduceArgs) -> Result<(), String> {
    let gen_ = load_generation(&a.gen_file)?;
    let shards = read_shards(&a)?;
    let mut ck = if a.state == "none" {
        fresh_checkpoint(&gen_, 0, initial_temperature(a.temperature, &gen_)?)
    } else {
        let s = std::fs::read_to_string(&a.state).map_err(|e| format!("{}: {e}", a.state))?;
        serde_json::from_str(&s).map_err(|e| format!("{}: {e}", a.state))?
    };
    if gen_hash(&ck.params) != gen_hash(&gen_.constants) {
        return Err(format!(
            "--state constants (gen_hash {}) differ from --gen constants ({})",
            gen_hash(&ck.params),
            gen_hash(&gen_.constants)
        ));
    }
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let step = StepOpts {
        lr: a.lr,
        learn_t: a.upd.learn_t(),
        conserve_mass: a.upd.conserve_mass,
    };
    let sum = reduce_round(&mut ck, &shards, step, &gen_.spec)?;
    write_round(&a.out_dir, &ck, &sum)?;
    print!("{}", sum.text);
    if !sum.finite {
        return Err("non-finite update".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use kepler_core::Position;
    use kepler_core::params::{SPEC_ID_1, spec_id};

    #[derive(Parser)]
    struct W {
        #[command(flatten)]
        a: TdleafArgs,
    }
    #[derive(Parser)]
    struct WS {
        #[command(flatten)]
        a: ShardArgs,
    }
    #[derive(Parser)]
    struct WR {
        #[command(flatten)]
        a: ReduceArgs,
    }

    impl PlayOpts {
        fn try_parse_from_args() -> PlayOpts {
            #[derive(Parser)]
            struct P {
                #[command(flatten)]
                p: PlayOpts,
            }
            P::try_parse_from(argv(&PLAY)).unwrap().p
        }
    }

    fn data(rel: &str) -> String {
        format!("{}/../../data/{rel}", env!("CARGO_MANIFEST_DIR"))
    }

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("kepler-train-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("x")
            .chain(v.iter().copied())
            .map(String::from)
            .collect()
    }

    const PLAY: [&str; 12] = [
        "--depth",
        "2",
        "--threads",
        "2",
        "--max-plies",
        "40",
        "--explore-plies",
        "2",
        "--lambda",
        "0.7",
        // keep the capped games in the gradient (the pre-T6 behaviour) for the round trip
        "--capped-target",
        "draw",
    ];

    #[test]
    fn two_in_process_shards_equal_two_shard_commands_plus_reduce() {
        let gen_f = data("generations/gen-000.json");
        let ops = data("openings/balanced-200.epd");
        let a_dir = tmp("inproc");
        let mut v = vec![
            "--gen",
            &gen_f,
            "--openings",
            &ops,
            "--rounds",
            "1",
            "--games-per-round",
            "6",
            "--shards",
            "2",
            "--seed",
            "5",
            "--out-dir",
            a_dir.to_str().unwrap(),
        ];
        v.extend(PLAY);
        run(W::try_parse_from(argv(&v)).unwrap().a).unwrap();
        // the round seed of round 1 with --seed 5
        let round_seed = XorShift::from_state(XorShift::new(5).state()).next_u64();
        let b_dir = tmp("sharded");
        let mut files = Vec::new();
        for k in 0..2u64 {
            let out = b_dir.join(format!("shard-{k}.json"));
            let (rs, ks) = (round_seed.to_string(), k.to_string());
            let mut v = vec![
                "--gen",
                &gen_f,
                "--openings",
                &ops,
                "--games",
                "3",
                "--seed",
                &rs,
                "--shard-index",
                &ks,
                "--out",
                out.to_str().unwrap(),
            ];
            v.extend(PLAY);
            run_shard(WS::try_parse_from(argv(&v)).unwrap().a).unwrap();
            files.push(out);
        }
        let out_dir = b_dir.join("out");
        let mut v = vec![
            "--gen",
            &gen_f,
            "--state",
            "none",
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--shards",
        ];
        let fs: Vec<String> = files
            .iter()
            .map(|f| f.to_str().unwrap().to_string())
            .collect();
        v.extend(fs.iter().map(String::as_str));
        run_reduce(WR::try_parse_from(argv(&v)).unwrap().a).unwrap();
        let read = |d: &Path, f: &str| -> serde_json::Value {
            serde_json::from_str(&std::fs::read_to_string(d.join(f)).unwrap()).unwrap()
        };
        let (ca, cb) = (
            read(&a_dir, "candidate.json"),
            read(&out_dir, "candidate.json"),
        );
        assert_eq!(ca["constants"], cb["constants"]);
        assert_eq!(ca["parent"], cb["parent"]);
        assert_eq!(ca["training"]["temperature"], cb["training"]["temperature"]);
        assert_ne!(ca["training"]["temperature"], json!(4.0), "T is learned");
        let moved = read(&a_dir, "round-001.json");
        assert_eq!(moved["games"], 6);
        assert_eq!(moved["games"], read(&out_dir, "round-001.json")["games"]);
        assert_ne!(
            ca["constants"],
            read(
                std::path::Path::new(&gen_f).parent().unwrap(),
                "gen-000.json"
            )["constants"],
            "the step must change something"
        );
        let _ = std::fs::remove_dir_all(a_dir);
        let _ = std::fs::remove_dir_all(b_dir);
    }

    fn step() -> StepOpts {
        StepOpts {
            lr: 0.01,
            learn_t: true,
            conserve_mass: false,
        }
    }

    /// A hand-made shard (no games): `grad_logt` and `plies` as given.
    fn synth_shard(p: &Params, t: f32, grad_logt: f64, plies: usize) -> Shard {
        Shard {
            kind: "tdleaf-shard".into(),
            version: 1,
            spec: SPEC_ID_1.into(),
            gen_hash: gen_hash(p),
            seed: 1,
            games: 1,
            plies,
            grad_sum: vec![0.0; NGRAD],
            grad_logt_sum: grad_logt,
            dropped_plies: 0,
            abs_d_sum: 0.0,
            d_sum: 0.0,
            results: ShardResults::default(),
            elapsed_ms: 1,
            config: json!({"lambda": 0.7, "temperature": t, "capped_target": "drop", "depth": 2}),
            deterministic: true,
            result_hash: None,
        }
    }

    #[test]
    fn temperature_derivative_matches_finite_difference() {
        // v = tanh(e / T) for a leaf; compare dv/dT with a central difference
        let p = Params::default();
        let pos = Position::from_fen("4k3/8/8/8/8/8/PPP5/RNB1K3 w - - 0 1").unwrap();
        for root_white in [true, false] {
            for t in [1.5f32, 4.0, 12.0] {
                let l = selfgame::leaf_value(&pos, root_white, &p, t);
                let h = 1e-2f32 * t;
                let up = selfgame::leaf_value(&pos, root_white, &p, t + h).v;
                let dn = selfgame::leaf_value(&pos, root_white, &p, t - h).v;
                let fd = (up - dn) / (2.0 * h);
                assert!(
                    (l.grad_t - fd).abs() < 2e-3 * (1.0 + fd.abs()),
                    "T {t}: analytic {} vs finite difference {fd}",
                    l.grad_t
                );
                assert!(l.grad_t.is_finite());
            }
        }
    }

    #[test]
    fn overconfident_predictions_raise_the_temperature() {
        // eval 8 ku at T = 4: v = tanh(2) = 0.964, but the mover then only draws (0):
        // every prediction is overconfident, so ascent on log T must raise T.
        let t = 4.0f32;
        let e = 8.0f32;
        let v = (e / t).tanh();
        let grad_t = -(1.0 - v * v) * e / (t * t);
        let leaves = vec![
            Leaf {
                v,
                grad: [0.0; NGRAD],
                grad_t
            };
            3
        ];
        let td = game_td(&leaves, 0.0, 0.0, t);
        assert!(td.grad_logt_sum > 0.0, "{}", td.grad_logt_sum);
        let mut adam = AdamT::default();
        let t1 = adam.step(t, td.grad_logt_sum / td.plies as f64, 0.01);
        assert!(t1 > t, "{t1}");
        // underconfident (eval 1 ku, then a win): T must fall
        let v = (1.0f32 / t).tanh();
        let grad_t = -(1.0 - v * v) / (t * t);
        let leaves = vec![Leaf {
            v,
            grad: [0.0; NGRAD],
            grad_t,
        }];
        let td = game_td(&leaves, 1.0, 0.0, t);
        assert!(td.grad_logt_sum < 0.0);
        let mut adam = AdamT::default();
        assert!(adam.step(t, td.grad_logt_sum, 0.01) < t);
        // bounds: a persistent push stays inside [0.5, 50]
        let mut adam = AdamT::default();
        let mut x = 49.9f32;
        for _ in 0..50 {
            x = adam.step(x, 1.0, 0.5);
        }
        assert!((x - 50.0).abs() < 1e-3, "{x}");
        let mut adam = AdamT::default();
        for _ in 0..50 {
            x = adam.step(x, -1.0, 0.5);
        }
        assert!((x - 0.5).abs() < 1e-3, "{x}");
    }

    #[test]
    fn reduce_learns_or_fixes_temperature_and_checks_it() {
        let gen_ = load_generation(Path::new(&data("generations/gen-000.json"))).unwrap();
        let sh = synth_shard(&gen_.constants, 4.0, 5.0, 10);
        let mut ck = fresh_checkpoint(&gen_, 1, 4.0);
        let s = reduce_round(&mut ck, std::slice::from_ref(&sh), step(), SPEC_ID_1).unwrap();
        assert!(ck.temperature > 4.0);
        assert_eq!(s.candidate.training["temperature"], json!(ck.temperature));
        assert_eq!(s.round_json["temperature"]["before"], json!(4.0));
        // fixed temperature
        let mut ck = fresh_checkpoint(&gen_, 1, 4.0);
        let fixed = StepOpts {
            learn_t: false,
            ..step()
        };
        reduce_round(&mut ck, std::slice::from_ref(&sh), fixed, SPEC_ID_1).unwrap();
        assert_eq!(ck.temperature, 4.0);
        // shards played at a different temperature are refused
        let mut ck = fresh_checkpoint(&gen_, 1, 5.0);
        let e = reduce_round(&mut ck, std::slice::from_ref(&sh), step(), SPEC_ID_1).unwrap_err();
        assert!(e.contains("temperature"), "{e}");
        // differing capped_target between shards is refused
        let mut other = sh.clone();
        other.config["capped_target"] = json!("draw");
        let mut ck = fresh_checkpoint(&gen_, 1, 4.0);
        let e = reduce_round(&mut ck, &[sh, other], step(), SPEC_ID_1).unwrap_err();
        assert!(e.contains("capped_target"), "{e}");
    }

    #[test]
    fn conserve_mass_keeps_the_total_and_the_ratios() {
        let mut p = Params::default();
        let target = total_mass(&p);
        assert_eq!(target, 39.0);
        // a step inflated every mass
        p.m_n = 3.6;
        p.m_b = 3.9;
        p.m_r = 6.1;
        p.m_q = 11.0;
        assert!(total_mass(&p) > target + 1.0);
        let ratio = |p: &Params| [p.m_b / p.m_n, p.m_r / p.m_n, p.m_q / p.m_n];
        let r0 = ratio(&p);
        conserve_mass(&mut p, target);
        assert!((total_mass(&p) - target).abs() < 1e-4, "{}", total_mass(&p));
        for (a, b) in r0.iter().zip(ratio(&p)) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // clamped masses may break the total, but never the bounds
        let mut q = Params {
            m_n: 1.0,
            ..Default::default()
        };
        conserve_mass(&mut q, 80.0);
        assert!(q.m_q <= 20.0 && q.m_r <= 12.0 && q.m_n <= 8.0 && q.m_b <= 8.0);
        // through the reduce step
        let gen_ = load_generation(Path::new(&data("generations/gen-000.json"))).unwrap();
        let mut ck = fresh_checkpoint(&gen_, 1, 4.0);
        let t0 = total_mass(&ck.params);
        let mut sh = synth_shard(&gen_.constants, 4.0, 0.0, 10);
        sh.grad_sum[3] = 50.0; // push m_q up hard
        let on = StepOpts {
            conserve_mass: true,
            ..step()
        };
        reduce_round(&mut ck, &[sh], on, SPEC_ID_1).unwrap();
        assert!((total_mass(&ck.params) - t0).abs() < 1e-3);
    }

    #[test]
    fn capped_target_flag_parses_and_drop_removes_the_gradient() {
        let o = PlayOpts::try_parse_from_args();
        assert_eq!(o.capped_target, CappedTarget::Draw);
        #[derive(Parser)]
        struct P {
            #[command(flatten)]
            p: PlayOpts,
        }
        let d = P::try_parse_from(argv(&[])).unwrap().p;
        assert_eq!(d.capped_target, CappedTarget::Drop);
        assert_eq!(d.temperature, None);
        let s = P::try_parse_from(argv(&["--capped-target", "score"]))
            .unwrap()
            .p;
        assert_eq!(s.capped_target, CappedTarget::Score);
        // tiny games that all hit a 6-ply cap
        let gen_ = load_generation(Path::new(&data("generations/gen-000.json"))).unwrap();
        let ops = load_openings(Path::new(&data("openings/balanced-200.epd"))).unwrap();
        let mut o = o;
        o.depth = 1;
        o.max_plies = 6;
        o.explore_plies = 2;
        o.capped_target = CappedTarget::Drop;
        let sh = play_shard(&gen_.constants, &ops, 2, 3, &o, 4.0).unwrap();
        assert_eq!(sh.games, 2);
        assert_eq!(sh.results.capped, 2);
        assert_eq!(sh.plies, 0, "dropped games add no plies");
        assert_eq!(sh.dropped_plies, 12);
        assert!(sh.grad_sum.iter().all(|&g| g == 0.0) && sh.grad_logt_sum == 0.0);
        assert_eq!(sh.config["capped_target"], "drop");
        o.capped_target = CappedTarget::Score;
        let sh = play_shard(&gen_.constants, &ops, 2, 3, &o, 4.0).unwrap();
        assert_eq!((sh.plies, sh.dropped_plies), (12, 0));
        assert_eq!(sh.config["temperature"], json!(4.0));
    }

    #[test]
    fn learn_tide_adds_the_thirteenth_gradient_entry() {
        let gen_ = load_generation(Path::new(&data("generations/gen-000.json"))).unwrap();
        let ops = load_openings(Path::new(&data("openings/balanced-200.epd"))).unwrap();
        let mut o = PlayOpts::try_parse_from_args();
        o.depth = 1;
        o.max_plies = 6;
        o.explore_plies = 2;
        o.capped_target = CappedTarget::Score;
        // default: a spec-1 shard (12 entries, no learn_tide key)
        let a = play_shard(&gen_.constants, &ops, 2, 3, &o, 4.0).unwrap();
        assert_eq!(a.grad_sum.len(), OLD_NGRAD);
        assert!(a.config.get("learn_tide").is_none());
        assert_eq!(a.spec, SPEC_ID_1);
        o.learn_tide = true;
        let b = play_shard(&gen_.constants, &ops, 2, 3, &o, 4.0).unwrap();
        assert_eq!(b.grad_sum.len(), NGRAD);
        assert_eq!(b.grad_sum[..OLD_NGRAD], a.grad_sum[..]);
        assert!(
            b.grad_sum[OLD_NGRAD] != 0.0,
            "gradient exists at lambda_tide = 0"
        );
        assert_eq!(b.config["learn_tide"], json!(true));
        // the round moves lambda_tide only for the 13-entry shard
        let step = StepOpts {
            lr: 0.01,
            learn_t: false,
            conserve_mass: false,
        };
        let mk = || Checkpoint {
            round: 0,
            games: 0,
            parent: 0,
            params: gen_.constants,
            adam: Adam::new(),
            rng_state: 0,
            bound_history: vec![],
            temperature: 4.0,
            adam_t: AdamT::default(),
            mass_total: 0.0,
        };
        let (mut c12, mut c13) = (mk(), mk());
        let r12 = reduce_round(&mut c12, &[a], step, SPEC_ID_1).unwrap();
        let r13 = reduce_round(&mut c13, &[b], step, SPEC_ID_1).unwrap();
        assert_eq!(c12.params.lambda_tide, 0.0);
        // (the step may be negative and then sits on the lower bound; the moment shows it)
        assert!(c13.adam.m[OLD_NGRAD] != 0.0 && c12.adam.m[OLD_NGRAD] == 0.0);
        assert_eq!(r12.candidate.spec, SPEC_ID_1);
        assert_eq!(r13.candidate.spec, spec_id(&c13.params));
        assert!(r12.round_json["grad_avg"].get("lambda_tide").is_none());
        assert!(r13.round_json["grad_avg"].get("lambda_tide").is_some());
    }

    #[test]
    fn reduce_refuses_mismatched_gen_hash() {
        let gen_ = load_generation(Path::new(&data("generations/gen-000.json"))).unwrap();
        let ops = load_openings(Path::new(&data("openings/balanced-200.epd"))).unwrap();
        let o = PlayOpts::try_parse_from_args();
        let s1 = play_shard(&gen_.constants, &ops, 1, 1, &o, 4.0).unwrap();
        let mut other = gen_.constants;
        other.alpha -= 0.05;
        let s2 = play_shard(&other, &ops, 1, 2, &o, 4.0).unwrap();
        assert_ne!(s1.gen_hash, s2.gen_hash);
        let mut ck = fresh_checkpoint(&gen_, 1, 4.0);
        let e = reduce_round(&mut ck, &[s1.clone(), s2.clone()], step(), SPEC_ID_1).unwrap_err();
        assert!(e.contains("gen_hash"), "{e}");
        // all shards agree with each other but not with the current constants
        let mut ck = fresh_checkpoint(&gen_, 1, 4.0);
        let e = reduce_round(&mut ck, &[s2], step(), SPEC_ID_1).unwrap_err();
        assert!(e.contains("gen_hash"), "{e}");
        assert_eq!(ck.round, 0);
        // matching shards are accepted
        assert!(reduce_round(&mut ck, &[s1], step(), SPEC_ID_1).is_ok());
    }

    #[test]
    fn td_and_lambda_return_by_hand() {
        // 3 plies: v = [0.2, -0.1, 0.4]; last mover wins (+1); lambda = 0.7
        // d0 = -v1 - v0 = 0.1 - 0.2 = -0.1
        // d1 = -v2 - v1 = -0.4 + 0.1 = -0.3
        // d2 = 1 - 0.4 = 0.6
        // e2 = 0.6; e1 = -0.3 - 0.7*0.6 = -0.72; e0 = -0.1 - 0.7*(-0.72) = 0.404
        let (d, e) = td_errors(&[0.2, -0.1, 0.4], 1.0, 0.7);
        for (x, y) in d.iter().zip([-0.1, -0.3, 0.6]) {
            assert!((x - y).abs() < 1e-6, "{d:?}");
        }
        for (x, y) in e.iter().zip([0.404, -0.72, 0.6]) {
            assert!((x - y).abs() < 1e-6, "{e:?}");
        }
        // lambda = 0: e = d
        let (d0, e0) = td_errors(&[0.2, -0.1, 0.4], 1.0, 0.0);
        assert_eq!(d0, e0);
        // loss for the last mover
        let (d, _) = td_errors(&[0.5], -1.0, 0.7);
        assert!((d[0] + 1.5).abs() < 1e-6);
    }

    #[test]
    fn sign_convention_ascent_raises_v_toward_a_win() {
        // v = tanh(theta) on a single ply that the mover then wins: v should rise.
        let theta = 0.2f32;
        let v = theta.tanh();
        let mut grad = [0.0f32; NGRAD];
        grad[5] = 1.0 - v * v; // slot 5 = alpha, range 1
        let td = game_td(
            &[Leaf {
                v,
                grad,
                grad_t: 0.0,
            }],
            1.0,
            0.7,
            4.0,
        );
        assert!(td.grad_sum[5] > 0.0);
        let mut p = Params::default();
        let mut adam = Adam::new();
        let a0 = p.alpha;
        let mut g = td.grad_sum;
        g.iter_mut().for_each(|x| *x /= td.plies as f32);
        p.set_grad_values(adam.step(p.grad_values(), g, 0.01));
        assert!(p.alpha > a0, "alpha {} -> {}", a0, p.alpha);
        // and the other way when the mover loses
        let td = game_td(
            &[Leaf {
                v,
                grad,
                grad_t: 0.0,
            }],
            -1.0,
            0.7,
            4.0,
        );
        let mut p = Params::default();
        let mut adam = Adam::new();
        p.set_grad_values(adam.step(p.grad_values(), td.grad_sum, 0.01));
        assert!(p.alpha < a0);
        // the new theta would predict better: v(theta + dtheta) is closer to +1
        assert!((theta + 0.01f32).tanh() > v);
    }

    #[test]
    fn spec1_checkpoint_and_constants_still_load() {
        // A 12-slot Adam state (written before lambda_tide) loads with a zero moment for it.
        let mut adam = Adam::new();
        for i in 0..OLD_NGRAD {
            adam.m[i] = 0.5 + i as f64;
            adam.v[i] = 0.25 + i as f64;
        }
        adam.t = 7;
        let mut j = serde_json::to_value(&adam).unwrap();
        j["m"].as_array_mut().unwrap().truncate(OLD_NGRAD);
        j["v"].as_array_mut().unwrap().truncate(OLD_NGRAD);
        let back: Adam = serde_json::from_value(j.clone()).unwrap();
        assert_eq!(back.m[..OLD_NGRAD], adam.m[..OLD_NGRAD]);
        assert_eq!(
            (back.m[OLD_NGRAD], back.v[OLD_NGRAD], back.t),
            (0.0, 0.0, 7)
        );
        j["m"].as_array_mut().unwrap().truncate(5);
        assert!(serde_json::from_value::<Adam>(j).is_err());
        // With lambda_tide = 0 the hashed constants have no lambda_tide key, and
        // switching it on changes the hash.
        let p = Params::default();
        assert!(constants_json(&p).get("lambda_tide").is_none());
        let q = Params {
            lambda_tide: 0.1,
            ..p
        };
        assert!(constants_json(&q).get("lambda_tide").is_some());
        assert_ne!(gen_hash(&p), gen_hash(&q));
        assert_eq!(spec_id(&p), SPEC_ID_1);
        assert!(at_bound(&p, false).iter().all(|n| n != "lambda_tide"));
        assert!(at_bound(&p, true).iter().any(|n| n == "lambda_tide"));
    }

    #[test]
    fn adam_step_and_projection() {
        // alpha (slot 5) has range 1, so normalised units equal raw units
        let mut adam = Adam::new();
        let mut th = Params::default().grad_values();
        let a0 = th[5];
        let mut g = [0.0f32; NGRAD];
        g[5] = 3.0;
        th = adam.step(th, g, 0.01);
        // step 1: mhat = g, vhat = g^2 -> step = lr * g/(|g| + eps) ~ lr
        assert!((th[5] - a0 - 0.01).abs() < 1e-6, "{}", th[5] - a0);
        // m_q (slot 3) has range 16: a zero gradient does not move it
        assert_eq!(th[3], 9.0);
        // step 2 with g = -3 (normalised -g1=1 pattern): m = 0.9*0.1*3 - 0.1*3 = -0.03,
        // mhat = -0.03/0.19, v = 0.999*0.001*9 + 0.001*9 = 0.017991, vhat = 9
        // step = 0.01 * (-0.157895)/3 = -5.2632e-4
        g[5] = -3.0;
        let a1 = th[5];
        th = adam.step(th, g, 0.01);
        assert!((th[5] - a1 + 5.2632e-4).abs() < 1e-6, "{}", th[5] - a1);
        // range scaling: the same normalised step moves m_q 16x further
        let mut adam = Adam::new();
        let mut g = [0.0f32; NGRAD];
        g[3] = 1.0;
        let t = adam.step(Params::default().grad_values(), g, 0.01);
        assert!((t[3] - 9.0 - 0.16).abs() < 1e-5);
        // projection onto bounds
        let mut p = Params::default();
        let mut big = p.grad_values();
        big[5] = 7.0; // alpha <= 1
        big[0] = -3.0; // m_n >= 1
        p.set_grad_values(big);
        assert_eq!(p.alpha, 1.0);
        assert_eq!(p.m_n, 1.0);
        assert_eq!(
            at_bound(&p, false),
            vec!["m_n".to_string(), "alpha".to_string()]
        );
    }

    #[test]
    fn pinned_needs_majority() {
        let a = vec!["alpha".to_string()];
        let h: Vec<Vec<String>> = (0..10)
            .map(|i| if i < 6 { a.clone() } else { vec![] })
            .collect();
        assert_eq!(pinned(&h), a);
        let h: Vec<Vec<String>> = (0..10)
            .map(|i| if i < 5 { a.clone() } else { vec![] })
            .collect();
        assert!(pinned(&h).is_empty());
    }
}
