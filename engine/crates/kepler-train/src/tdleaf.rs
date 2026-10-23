//! `kepler-train tdleaf`: TDLeaf(lambda) rounds (`docs/v1/04_TRAINING_AND_GATES.md` §1).
//!
//! Conventions
//! * `v_t = tanh(eval_white(leaf_t) / s)` in the view of the side that moved at ply `t`;
//!   a mate score gives `v_t = +-1` with zero gradient.
//! * `d_t = -v_{t+1} - v_t` (next value negated into the same view); the last ply uses the
//!   game result for its mover (+1 / 0 / -1; the max-plies cap counts as 0).
//! * Update direction `sum_t grad v_t * e_t`, `e_t = d_t - lambda * e_{t+1}`. The spec's
//!   `sum_{j>=t} lambda^{j-t} d_j` is written with every `d_j` in one common view; with
//!   `d_j` in its own mover's view, ply `j`'s error enters ply `t`'s sum with sign
//!   `(-1)^{j-t}`, which is what the recursion implements. (With lambda = 0 the two agree.)
//! * Gradient ASCENT on agreement: `v` moves toward its target, so theta += step.
//! * Adam runs on theta normalised by its bound range `hi - lo`, so `--lr 0.01` means
//!   "about 1% of a constant's allowed range per round" for all 12 constants alike. The
//!   result is projected onto the bounds.
//! * `c` and `f_acc` are left unchanged (outer-loop constants).
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
use clap::Args;
use kepler_core::params::{Generation, SPEC_ID};
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
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long, default_value_t = 0.7)]
    lambda: f32,
    /// The s in tanh(eval / s), ku
    #[arg(long, default_value_t = 4.0)]
    scale: f32,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 8)]
    explore_plies: usize,
    /// Transposition table per searcher, MB
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[command(flatten)]
    adj: AdjArgs,
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
    pub abs_d_sum: f64,
    pub d_sum: f64,
    pub results: ShardResults,
    pub elapsed_ms: u64,
    pub config: serde_json::Value,
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
) -> Result<Shard, String> {
    let cfg = GameCfg {
        limits: Limits {
            depth: Some(o.depth),
            ..Default::default()
        },
        explore_depth: o.depth.saturating_sub(1).max(1),
        max_plies: o.max_plies,
        explore_plies: o.explore_plies,
        explore_margin: 0.15,
        record: true,
        scale: o.scale,
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
            let td = game_td(&g.leaves, g.result_for_mover(last), o.lambda);
            Ok::<_, String>(Some((td, g.result, g.capped, g.adjudicated)))
        },
    );
    let mut grad = vec![0.0f64; NGRAD];
    let mut sh = Shard {
        kind: "tdleaf-shard".into(),
        version: 1,
        spec: SPEC_ID.into(),
        gen_hash: gen_hash(params),
        seed,
        games: 0,
        plies: 0,
        grad_sum: Vec::new(),
        abs_d_sum: 0.0,
        d_sum: 0.0,
        results: ShardResults::default(),
        elapsed_ms: 0,
        config: serde_json::Value::Null,
    };
    for g in played {
        let Some((td, res, cap, adj)) = g? else {
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
        sh.plies += td.plies;
        sh.abs_d_sum += td.sum_abs_d;
        sh.d_sum += td.sum_d;
        for i in 0..NGRAD {
            grad[i] += td.grad_sum[i] as f64;
        }
    }
    sh.grad_sum = grad;
    sh.elapsed_ms = t0.elapsed().as_millis() as u64;
    sh.config = json!({
        "depth": o.depth, "threads": o.threads, "hash_mb": o.hash,
        "lambda": o.lambda, "scale": o.scale,
        "max_plies": o.max_plies, "explore_plies": o.explore_plies, "explore_margin": 0.15,
        "games_requested": games,
        "adjudication": o.adj.cfg(),
        "created": util::utc_iso(),
    });
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
    pub plies: usize,
    pub sum_abs_d: f64,
    pub sum_d: f64,
}

pub fn game_td(leaves: &[Leaf], result_last: f32, lambda: f32) -> GameTd {
    let vs: Vec<f32> = leaves.iter().map(|l| l.v).collect();
    let (d, e) = td_errors(&vs, result_last, lambda);
    let mut grad_sum = [0.0f32; NGRAD];
    for (l, et) in leaves.iter().zip(&e) {
        for i in 0..NGRAD {
            grad_sum[i] += l.grad[i] * et;
        }
    }
    GameTd {
        grad_sum,
        plies: leaves.len(),
        sum_abs_d: d.iter().map(|x| x.abs() as f64).sum(),
        sum_d: d.iter().map(|&x| x as f64).sum(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adam {
    pub m: [f64; NGRAD],
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
            1.0 - BETA1.powi(self.t as i32),
            1.0 - BETA2.powi(self.t as i32),
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

/// Constants sitting on a bound.
pub fn at_bound(p: &Params) -> Vec<String> {
    let mut v = Vec::new();
    for (i, x) in p.grad_values().iter().enumerate() {
        let (lo, hi) = Params::GRAD_BOUNDS[i];
        let tol = 1e-6 * (hi - lo);
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

fn num(v: &serde_json::Value, k: &str) -> Option<f64> {
    v.get("config")?.get(k)?.as_f64()
}

/// Validate `shards`, sum them, and take one Adam step on `ck` (which is advanced).
/// Shards must share spec and gen_hash, and that hash must be the hash of `ck.params`
/// (the constants the shards must have been played with).
pub fn reduce_round(
    ck: &mut Checkpoint,
    shards: &[Shard],
    lr: f32,
    spec: &str,
) -> Result<RoundSummary, String> {
    let first = shards.first().ok_or("no shards to reduce")?;
    let want = gen_hash(&ck.params);
    for (i, s) in shards.iter().enumerate() {
        if s.kind != "tdleaf-shard" || s.version != 1 {
            return Err(format!("shard {i}: not a tdleaf-shard v1 file"));
        }
        if s.spec != first.spec || s.spec != SPEC_ID || spec != SPEC_ID {
            return Err(format!(
                "shard {i}: spec {} differs from {} / generation {spec} / engine {SPEC_ID}",
                s.spec, first.spec
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
        if s.grad_sum.len() != NGRAD {
            return Err(format!(
                "shard {i}: grad_sum has {} entries",
                s.grad_sum.len()
            ));
        }
        for k in ["lambda", "scale"] {
            if num(&s.config, k) != num(&first.config, k) {
                return Err(format!("shard {i}: config {k} differs from shard 0"));
            }
        }
    }
    let mut grad = [0.0f64; NGRAD];
    let (mut plies, mut games, mut sum_abs_d, mut sum_d, mut ms) = (0usize, 0usize, 0.0, 0.0, 0u64);
    let mut res = ShardResults::default();
    for s in shards {
        plies += s.plies;
        games += s.games;
        sum_abs_d += s.abs_d_sum;
        sum_d += s.d_sum;
        ms += s.elapsed_ms;
        res.w += s.results.w;
        res.d += s.results.d;
        res.l += s.results.l;
        res.capped += s.results.capped;
        res.adjudicated += s.results.adjudicated;
        for i in 0..NGRAD {
            grad[i] += s.grad_sum[i];
        }
    }
    let np = plies.max(1) as f64;
    let mut g32 = [0.0f32; NGRAD];
    for i in 0..NGRAD {
        g32[i] = (grad[i] / np) as f32;
    }
    let before = ck.params.grad_values();
    let proposed = ck.adam.step(before, g32, lr);
    let mut next = ck.params;
    next.set_grad_values(proposed);
    let after = next.grad_values();
    let delta: Vec<f32> = (0..NGRAD).map(|i| after[i] - before[i]).collect();
    ck.params = next;
    ck.round += 1;
    ck.games += games;
    let bound = at_bound(&ck.params);
    ck.bound_history.push(bound.clone());
    let pin = pinned(&ck.bound_history);
    let finite = g32.iter().all(|x| x.is_finite()) && after.iter().all(|x| x.is_finite());
    let secs = ms as f64 / 1000.0;
    let (depth, lambda, scale) = (
        first.config.get("depth").cloned().unwrap_or_default(),
        first.config.get("lambda").cloned().unwrap_or_default(),
        first.config.get("scale").cloned().unwrap_or_default(),
    );
    let mut names_delta = serde_json::Map::new();
    let mut names_grad = serde_json::Map::new();
    for i in 0..NGRAD {
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
        "depth": depth, "lambda": lambda, "scale": scale, "lr": lr,
        "adjudication": first.config.get("adjudication").cloned().unwrap_or_default(),
        "finite": finite,
    });
    let candidate = Generation {
        generation: ck.parent + 1,
        parent: Some(ck.parent),
        spec: spec.to_string(),
        created: util::utc_date(),
        constants: ck.params,
        pinned: pin.clone(),
        training: json!({
            "rounds": ck.round, "games": ck.games, "depth": depth,
            "lambda": lambda, "scale": scale, "lr": lr,
            "lr_units": "relative to bound range (hi - lo)", "optimizer": "adam",
            "games_per_round": games, "shards_per_round": shards.len(),
        }),
        acceptance: serde_json::Value::Null,
    };
    let mut text = format!(
        "round {} | {} games from {} shards ({:.2}/s, {:.0}s of shard time) | W{} D{} L{} cap{} adj{} | mean plies {:.1} | mean|d| {:.4} mean d {:+.4} | finite {}\n",
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
        finite
    );
    for i in 0..NGRAD {
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

fn write_json(path: PathBuf, v: &serde_json::Value) -> Result<(), String> {
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

fn fresh_checkpoint(parent: &Generation, seed: u64) -> Checkpoint {
    Checkpoint {
        round: 0,
        games: 0,
        parent: parent.generation,
        params: parent.constants,
        adam: Adam::new(),
        rng_state: XorShift::new(seed).state(),
        bound_history: Vec::new(),
    }
}

pub fn run(a: TdleafArgs) -> Result<(), String> {
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let parent = load_generation(&a.gen_file)?;
    let openings = load_openings(&a.openings)?;
    if a.shards == 0 || a.games_per_round < a.shards {
        return Err("--shards must be >= 1 and <= --games-per-round".into());
    }
    let mut ck = fresh_checkpoint(&parent, a.seed);
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
            )?;
            sh.config["round_seed"] = json!(round_seed);
            sh.config["shard_index"] = json!(k);
            shards.push(sh);
        }
        let sum = reduce_round(&mut ck, &shards, a.lr, &parent.spec)?;
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
    let mut sh = play_shard(&gen_.constants, &openings, a.games, seed, &a.play)?;
    sh.config["generation"] = json!(gen_.generation);
    sh.config["gen_file"] = json!(a.gen_file);
    sh.config["openings"] = json!(a.openings);
    if let Some(k) = a.shard_index {
        sh.config["round_seed"] = json!(a.seed);
        sh.config["shard_index"] = json!(k);
    }
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
    files
        .iter()
        .map(|p| {
            let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
        })
        .collect()
}

pub fn run_reduce(a: ReduceArgs) -> Result<(), String> {
    let gen_ = load_generation(&a.gen_file)?;
    let shards = read_shards(&a)?;
    let mut ck = if a.state == "none" {
        fresh_checkpoint(&gen_, 0)
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
    let sum = reduce_round(&mut ck, &shards, a.lr, &gen_.spec)?;
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

    const PLAY: [&str; 10] = [
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

    #[test]
    fn reduce_refuses_mismatched_gen_hash() {
        let gen_ = load_generation(Path::new(&data("generations/gen-000.json"))).unwrap();
        let ops = load_openings(Path::new(&data("openings/balanced-200.epd"))).unwrap();
        let o = PlayOpts::try_parse_from_args();
        let s1 = play_shard(&gen_.constants, &ops, 1, 1, &o).unwrap();
        let mut other = gen_.constants;
        other.alpha -= 0.05;
        let s2 = play_shard(&other, &ops, 1, 2, &o).unwrap();
        assert_ne!(s1.gen_hash, s2.gen_hash);
        let mut ck = fresh_checkpoint(&gen_, 1);
        let e = reduce_round(&mut ck, &[s1.clone(), s2.clone()], 0.01, SPEC_ID).unwrap_err();
        assert!(e.contains("gen_hash"), "{e}");
        // all shards agree with each other but not with the current constants
        let mut ck = fresh_checkpoint(&gen_, 1);
        let e = reduce_round(&mut ck, &[s2], 0.01, SPEC_ID).unwrap_err();
        assert!(e.contains("gen_hash"), "{e}");
        assert_eq!(ck.round, 0);
        // matching shards are accepted
        assert!(reduce_round(&mut ck, &[s1], 0.01, SPEC_ID).is_ok());
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
        let td = game_td(&[Leaf { v, grad }], 1.0, 0.7);
        assert!(td.grad_sum[5] > 0.0);
        let mut p = Params::default();
        let mut adam = Adam::new();
        let a0 = p.alpha;
        let mut g = td.grad_sum;
        g.iter_mut().for_each(|x| *x /= td.plies as f32);
        p.set_grad_values(adam.step(p.grad_values(), g, 0.01));
        assert!(p.alpha > a0, "alpha {} -> {}", a0, p.alpha);
        // and the other way when the mover loses
        let td = game_td(&[Leaf { v, grad }], -1.0, 0.7);
        let mut p = Params::default();
        let mut adam = Adam::new();
        p.set_grad_values(adam.step(p.grad_values(), td.grad_sum, 0.01));
        assert!(p.alpha < a0);
        // the new theta would predict better: v(theta + dtheta) is closer to +1
        assert!((theta + 0.01f32).tanh() > v);
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
        assert_eq!(at_bound(&p), vec!["m_n".to_string(), "alpha".to_string()]);
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
