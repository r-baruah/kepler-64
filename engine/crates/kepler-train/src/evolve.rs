//! `kepler-train evolve`, `evolve-shard`, `evolve-reduce`: cosmological natural selection
//! (T3) with a cooling schedule (T4) (`docs/v1/10_UNIVERSE_METHODS.md`).
//!
//! Twin universes `theta+` and `theta-` around the current constants play each other; the
//! winner pulls the constants towards itself. This is SPSA on playing strength.
//!
//! * Parameters: the 12 `GRAD_NAMES` constants plus `f_acc` (13; `c` is discrete and stays).
//!   Perturbation and step are in *relative* units: coordinate `i` moves in units of
//!   `range_i = hi - lo` (`f_acc`: 1).
//! * Iteration `K` (0-based). Cooling: `c_K = c0 / (K+1)^0.101`, `a_K = a0 / (K+1+A)^0.602`.
//! * Shard `k` of iteration `K` with round seed `S`: draw `Delta` (Rademacher, +-1 per
//!   parameter) from `XorShift::new(mix(mix(S, K), k))`, set
//!   `theta+-_i = clamp(theta_i +- c_K * range_i * Delta_i)`, play `--pairs` opening pairs
//!   `theta+` vs `theta-` (both colours per opening), report `score_plus` (the fraction of
//!   points `theta+` took, 0..1). Games and the opening order use `mix(mix(mix(S, K), k), 1)`.
//! * Reduce: per shard `g_i = (2 score_plus - 1) / (2 c_K Delta_i)`, averaged over shards;
//!   ascent `theta_i += a_K * range_i * g_i`, clamped to the bounds.
//! * State (`evolve-checkpoint-KKK.json`): the next iteration index, `a0`, `A`, `c0`, the
//!   current constants and the history of theta and mean `score_plus` per iteration.

#![allow(clippy::needless_range_loop)]

use crate::selfgame::{
    AdjArgs, GameCfg, Opening, gen_hash, load_generation, load_openings, par_run,
};
use crate::sprt::{Wdl, limits_from, play_pair};
use crate::tdleaf::{at_bound, pinned, write_json};
use crate::util::{self, XorShift};
use clap::Args;
use kepler_core::Params;
use kepler_core::params::{Generation, SPEC_ID};
use kepler_core::search::Searcher;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Number of evolved parameters: the 12 gradient constants and `f_acc`.
pub const NP: usize = 13;
/// SPSA exponents (Spall's standard choices).
pub const A_EXP: f64 = 0.602;
pub const C_EXP: f64 = 0.101;
pub const DEFAULT_A0: f64 = 0.1;
pub const DEFAULT_BIG_A: f64 = 10.0;
pub const DEFAULT_C0: f64 = 0.05;
pub const METHOD: &str = "cosmological-natural-selection-spsa";

pub fn name(i: usize) -> &'static str {
    if i < 12 {
        Params::GRAD_NAMES[i]
    } else {
        "f_acc"
    }
}

pub fn bounds(i: usize) -> (f32, f32) {
    if i < 12 {
        Params::GRAD_BOUNDS[i]
    } else {
        (0.0, 1.0)
    }
}

pub fn range(i: usize) -> f64 {
    let (lo, hi) = bounds(i);
    (hi - lo) as f64
}

pub fn theta_of(p: &Params) -> [f32; NP] {
    let mut t = [0.0; NP];
    t[..12].copy_from_slice(&p.grad_values());
    t[12] = p.f_acc;
    t
}

/// `base` with its 13 evolved parameters replaced by `th`, projected onto the bounds.
pub fn with_theta(base: &Params, th: &[f32; NP]) -> Params {
    let mut p = *base;
    let mut g = [0.0f32; 12];
    g.copy_from_slice(&th[..12]);
    p.set_grad_values(g);
    p.f_acc = th[12].clamp(0.0, 1.0);
    p
}

/// T4: perturbation size `c_K = c0 / (K+1)^0.101`.
pub fn c_k(c0: f64, k: u64) -> f64 {
    c0 / (k as f64 + 1.0).powf(C_EXP)
}

/// T4: step size `a_K = a0 / (K+1+A)^0.602`.
pub fn a_k(a0: f64, big_a: f64, k: u64) -> f64 {
    a0 / (k as f64 + 1.0 + big_a).powf(A_EXP)
}

/// Seed of shard `k` of iteration `iter` under round seed `seed`.
pub fn shard_seed(seed: u64, iter: u64, k: u64) -> u64 {
    util::mix(util::mix(seed, iter), k)
}

/// The Rademacher vector of a shard (+1 / -1 per parameter).
pub fn derive_delta(seed: u64, iter: u64, k: u64) -> [i8; NP] {
    let mut rng = XorShift::new(shard_seed(seed, iter, k));
    let mut d = [1i8; NP];
    for x in d.iter_mut() {
        *x = if rng.next_u64() >> 63 == 1 { 1 } else { -1 };
    }
    d
}

/// The twin universes `(theta+, theta-)` around `centre`.
pub fn twins(centre: &Params, delta: &[i8; NP], ck: f64) -> (Params, Params) {
    let th = theta_of(centre);
    let (mut plus, mut minus) = (th, th);
    for i in 0..NP {
        let d = ck * range(i) * delta[i] as f64;
        plus[i] = (th[i] as f64 + d) as f32;
        minus[i] = (th[i] as f64 - d) as f32;
    }
    (with_theta(centre, &plus), with_theta(centre, &minus))
}

/// Game options shared by `evolve` and `evolve-shard`.
#[derive(Args, Debug, Clone)]
pub struct EvolvePlay {
    #[arg(long)]
    depth: Option<u32>,
    /// Milliseconds per move (instead of --depth)
    #[arg(long)]
    movetime: Option<u64>,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 6)]
    explore_plies: usize,
    /// Transposition table per searcher, MB
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[command(flatten)]
    adj: AdjArgs,
}

#[derive(Args, Debug)]
pub struct ShardArgs {
    /// The centre generation (its constants are theta)
    #[arg(long = "gen")]
    gen_file: PathBuf,
    /// Evolve checkpoint (supplies c0 and the expected iteration), or "none"
    #[arg(long, default_value = "none")]
    state: String,
    #[arg(long)]
    openings: PathBuf,
    /// Iteration K (0-based; with --state it must equal the checkpoint's iter)
    #[arg(long = "iter", default_value_t = 0)]
    iter: u64,
    /// Round seed S; the shard's Delta comes from mix(mix(S, K), shard-index)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long = "shard-index", default_value_t = 0)]
    shard_index: u64,
    /// Opening pairs (2 games each)
    #[arg(long, default_value_t = 50)]
    pairs: usize,
    /// Initial perturbation size c0 (default: the state's, else 0.05)
    #[arg(long)]
    c0: Option<f64>,
    #[command(flatten)]
    play: EvolvePlay,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args, Debug)]
pub struct ReduceArgs {
    /// Previous evolve-checkpoint-KKK.json, or "none" to start from --gen
    #[arg(long, default_value = "none")]
    state: String,
    /// The centre the shards were played around; its constants must match their centre_hash
    #[arg(long = "gen")]
    gen_file: PathBuf,
    #[arg(long, num_args = 1.., conflicts_with = "shard_dir")]
    shards: Vec<PathBuf>,
    /// Directory of shard *.json files (read in file-name order)
    #[arg(long = "shard-dir")]
    shard_dir: Option<PathBuf>,
    /// Initial step size a0 (default: the state's, else 0.1)
    #[arg(long)]
    a0: Option<f64>,
    /// Stability constant A (default: the state's, else 10)
    #[arg(long = "A")]
    big_a: Option<f64>,
    /// Initial perturbation size c0 (default: the state's, else 0.05)
    #[arg(long)]
    c0: Option<f64>,
    #[arg(long = "out-dir")]
    out_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct EvolveArgs {
    /// Starting generation (the first centre, parent of the candidate)
    #[arg(long = "gen")]
    gen_file: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long = "out-dir")]
    out_dir: PathBuf,
    /// Target total number of iterations (with --resume, counting those already done)
    #[arg(long, default_value_t = 1)]
    iters: u64,
    /// In-process shards per iteration (played one after another, each with all threads)
    #[arg(long, default_value_t = 1)]
    shards: u64,
    /// Opening pairs per shard
    #[arg(long, default_value_t = 50)]
    pairs: usize,
    #[arg(long)]
    a0: Option<f64>,
    #[arg(long = "A")]
    big_a: Option<f64>,
    #[arg(long)]
    c0: Option<f64>,
    #[arg(long = "max-minutes")]
    max_minutes: Option<f64>,
    #[arg(long)]
    resume: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[command(flatten)]
    play: EvolvePlay,
}

/// One work unit (JSON file written by `evolve-shard`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolveShard {
    pub kind: String,
    pub version: u32,
    pub spec: String,
    pub centre_hash: String,
    pub iter: u64,
    pub seed: u64,
    pub shard_index: u64,
    pub c_k: f64,
    pub delta: Vec<i8>,
    /// Points theta+ took per game (0..1).
    pub score_plus: f64,
    /// Pairs scoring 0, 0.5, 1, 1.5, 2 for theta+.
    pub pentanomial: [u32; 5],
    pub wdl: Wdl,
    pub elapsed_ms: u64,
    pub config: serde_json::Value,
}

/// Play one shard: `pairs` opening pairs of `theta+` against `theta-`.
#[allow(clippy::too_many_arguments)]
pub fn play_evolve_shard(
    centre: &Params,
    openings: &[Opening],
    iter: u64,
    seed: u64,
    k: u64,
    pairs: usize,
    c0: f64,
    o: &EvolvePlay,
) -> Result<EvolveShard, String> {
    if pairs == 0 {
        return Err("--pairs must be >= 1".into());
    }
    let limits = limits_from(o.depth, o.movetime)?;
    let delta = derive_delta(seed, iter, k);
    let ck = c_k(c0, iter);
    let (plus, minus) = twins(centre, &delta, ck);
    let gseed = util::mix(shard_seed(seed, iter, k), 1);
    let cfg = GameCfg {
        limits,
        explore_depth: o.depth.map_or(3, |d| d.saturating_sub(1).max(1)),
        max_plies: o.max_plies,
        explore_plies: o.explore_plies,
        explore_margin: 0.15,
        record: false,
        quiet_leaf: false,
        temperature: 4.0,
        adj: o.adj.cfg(),
    };
    let mut order: Vec<usize> = (0..openings.len()).collect();
    let mut rng = XorShift::new(gseed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let t0 = Instant::now();
    let played = par_run(
        pairs,
        o.threads,
        || [Searcher::new(o.hash), Searcher::new(o.hash)],
        |st, i| {
            let op = &openings[order[i % order.len()]];
            play_pair(&cfg, op, gseed, i, &plus, &minus, st)
        },
    );
    let mut penta = [0u32; 5];
    let mut wdl = [0u32; 3];
    let mut pts = 0.0;
    for p in played {
        let (x, w) = p?;
        pts += x;
        penta[((x * 2.0).round() as usize).min(4)] += 1;
        for i in 0..3 {
            wdl[i] += w[i];
        }
    }
    Ok(EvolveShard {
        kind: "evolve-shard".into(),
        version: 1,
        spec: SPEC_ID.into(),
        centre_hash: gen_hash(centre),
        iter,
        seed,
        shard_index: k,
        c_k: ck,
        delta: delta.to_vec(),
        score_plus: pts / (2.0 * pairs as f64),
        pentanomial: penta,
        wdl: Wdl {
            w: wdl[0],
            d: wdl[1],
            l: wdl[2],
        },
        elapsed_ms: t0.elapsed().as_millis() as u64,
        config: json!({
            "depth": o.depth, "movetime_ms": o.movetime, "pairs": pairs, "c0": c0,
            "threads": o.threads, "hash_mb": o.hash,
            "max_plies": o.max_plies, "explore_plies": o.explore_plies,
            "adjudication": o.adj.cfg(), "created": util::utc_iso(),
            "plus_hash": gen_hash(&plus), "minus_hash": gen_hash(&minus),
            "note": "pentanomial = pairs scoring 0, 0.5, 1, 1.5, 2 for theta+",
        }),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistEntry {
    pub iter: u64,
    /// The 13 parameters after this iteration's step.
    pub theta: Vec<f32>,
    pub mean_score_plus: f64,
    pub c_k: f64,
    pub a_k: f64,
}

/// Evolve checkpoint (`evolve-checkpoint-KKK.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolveState {
    /// Iterations completed = the index K of the next iteration.
    pub iter: u64,
    pub a0: f64,
    #[serde(rename = "A")]
    pub big_a: f64,
    pub c0: f64,
    pub parent: u32,
    pub params: Params,
    pub games: usize,
    pub history: Vec<HistEntry>,
    pub bound_history: Vec<Vec<String>>,
}

pub fn fresh_state(parent: &Generation, a0: f64, big_a: f64, c0: f64) -> EvolveState {
    EvolveState {
        iter: 0,
        a0,
        big_a,
        c0,
        parent: parent.generation,
        params: parent.constants,
        games: 0,
        history: Vec::new(),
        bound_history: Vec::new(),
    }
}

/// Explicit flag, else the state's value, else the default.
fn resolve(explicit: Option<f64>, state: Option<f64>, default: f64) -> Result<f64, String> {
    let v = explicit.or(state).unwrap_or(default);
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("invalid schedule constant {v}"))
    }
}

fn load_state(path: &str) -> Result<Option<EvolveState>, String> {
    if path == "none" {
        return Ok(None);
    }
    let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&s)
        .map(Some)
        .map_err(|e| format!("{path}: {e}"))
}

#[derive(Debug)]
pub struct IterSummary {
    pub iter_json: serde_json::Value,
    pub candidate: Generation,
    pub text: String,
}

/// Validate `shards`, average their SPSA gradients and take one step on `st` (advanced).
pub fn reduce_iter(
    st: &mut EvolveState,
    shards: &[EvolveShard],
    spec: &str,
) -> Result<IterSummary, String> {
    let first = shards.first().ok_or("no shards to reduce")?;
    let want = gen_hash(&st.params);
    let k = st.iter;
    let ck = c_k(st.c0, k);
    let mut seen = std::collections::BTreeSet::new();
    for (i, s) in shards.iter().enumerate() {
        if s.kind != "evolve-shard" || s.version != 1 {
            return Err(format!("shard {i}: not an evolve-shard v1 file"));
        }
        if s.spec != first.spec || s.spec != SPEC_ID || spec != SPEC_ID {
            return Err(format!(
                "shard {i}: spec {} differs from {} / generation {spec} / engine {SPEC_ID}",
                s.spec, first.spec
            ));
        }
        if s.centre_hash != first.centre_hash {
            return Err(format!(
                "shard {i}: centre_hash {} differs from shard 0 ({})",
                s.centre_hash, first.centre_hash
            ));
        }
        if s.centre_hash != want {
            return Err(format!(
                "shard {i}: centre_hash {} does not match the current centre ({want})",
                s.centre_hash
            ));
        }
        if s.iter != first.iter {
            return Err(format!(
                "shard {i}: iter {} differs from shard 0 ({})",
                s.iter, first.iter
            ));
        }
        if s.iter != k {
            return Err(format!(
                "shard {i}: iter {} but the state expects iteration {k}",
                s.iter
            ));
        }
        if s.seed != first.seed {
            return Err(format!("shard {i}: round seed differs from shard 0"));
        }
        if !seen.insert(s.shard_index) {
            return Err(format!(
                "shard {i}: duplicate shard_index {}",
                s.shard_index
            ));
        }
        if (s.c_k - ck).abs() > 1e-12 * ck.abs().max(1.0) {
            return Err(format!(
                "shard {i}: c_k {} does not match the schedule c0 = {}, K = {k} ({ck})",
                s.c_k, st.c0
            ));
        }
        let want_delta = derive_delta(s.seed, s.iter, s.shard_index);
        if s.delta != want_delta {
            return Err(format!(
                "shard {i}: delta does not match mix(mix(seed, iter), shard_index)"
            ));
        }
        if !s.score_plus.is_finite() {
            return Err(format!("shard {i}: score_plus is not finite"));
        }
        for key in ["depth", "movetime_ms", "pairs"] {
            if s.config.get(key) != first.config.get(key) {
                return Err(format!("shard {i}: config {key} differs from shard 0"));
            }
        }
    }
    let n = shards.len() as f64;
    let mut ghat = [0.0f64; NP];
    let mut mean_score = 0.0;
    let (mut ms, mut pairs) = (0u64, 0usize);
    let mut wdl = [0u32; 3];
    let mut penta = [0u32; 5];
    for s in shards {
        mean_score += s.score_plus / n;
        ms += s.elapsed_ms;
        pairs += s.pentanomial.iter().sum::<u32>() as usize;
        wdl[0] += s.wdl.w;
        wdl[1] += s.wdl.d;
        wdl[2] += s.wdl.l;
        for j in 0..5 {
            penta[j] += s.pentanomial[j];
        }
        for i in 0..NP {
            ghat[i] += (2.0 * s.score_plus - 1.0) / (2.0 * s.c_k * s.delta[i] as f64) / n;
        }
    }
    let ak = a_k(st.a0, st.big_a, k);
    let before = theta_of(&st.params);
    let mut proposed = before;
    for i in 0..NP {
        proposed[i] = (before[i] as f64 + ak * range(i) * ghat[i]) as f32;
    }
    st.params = with_theta(&st.params, &proposed);
    let after = theta_of(&st.params);
    let finite = after.iter().all(|x| x.is_finite()) && ghat.iter().all(|x| x.is_finite());
    st.iter += 1;
    st.games += 2 * pairs;
    st.history.push(HistEntry {
        iter: k,
        theta: after.to_vec(),
        mean_score_plus: mean_score,
        c_k: ck,
        a_k: ak,
    });
    let bound = at_bound(&st.params);
    st.bound_history.push(bound.clone());
    let pin = pinned(&st.bound_history);
    let mut names_g = serde_json::Map::new();
    let mut names_d = serde_json::Map::new();
    for i in 0..NP {
        names_g.insert(name(i).into(), json!(ghat[i]));
        names_d.insert(name(i).into(), json!(after[i] - before[i]));
    }
    let secs = ms as f64 / 1000.0;
    let iter_json = json!({
        "iter": k, "c_k": ck, "a_k": ak,
        "mean_score_plus": mean_score,
        "shards": shards.iter().map(|s| json!({
            "shard_index": s.shard_index, "score_plus": s.score_plus,
            "pentanomial": s.pentanomial, "elapsed_ms": s.elapsed_ms,
        })).collect::<Vec<_>>(),
        "pairs": pairs, "games": 2 * pairs, "pentanomial": penta,
        "wdl_plus": { "w": wdl[0], "d": wdl[1], "l": wdl[2] },
        "centre_hash": want,
        "theta_before": before.to_vec(), "theta_after": after.to_vec(),
        "ghat": names_g, "delta": names_d,
        "constants": crate::selfgame::constants_json(&st.params),
        "at_bound": bound, "pinned": pin,
        "a0": st.a0, "A": st.big_a, "c0": st.c0,
        "seed": first.seed, "finite": finite,
    });
    let candidate = Generation {
        generation: st.parent + 1,
        parent: Some(st.parent),
        spec: spec.to_string(),
        created: util::utc_date(),
        constants: st.params,
        pinned: pin.clone(),
        training: json!({
            "method": METHOD,
            "iters": st.iter, "games": st.games,
            "a0": st.a0, "A": st.big_a, "c0": st.c0,
            "a_exponent": A_EXP, "c_exponent": C_EXP,
            "shards_per_iter": shards.len(), "pairs_per_shard": first.config.get("pairs"),
            "depth": first.config.get("depth"), "movetime_ms": first.config.get("movetime_ms"),
            "last_mean_score_plus": mean_score,
            "params": (0..NP).map(name).collect::<Vec<_>>(),
            "units": "perturbation and step relative to bound range (hi - lo)",
        }),
        acceptance: serde_json::Value::Null,
    };
    let mut text = format!(
        "iter {k} | {} pairs from {} shards ({:.0}s of shard time) | mean score+ {:.4} | c_K {:.4} a_K {:.4} | finite {finite}\n",
        pairs,
        shards.len(),
        secs,
        mean_score,
        ck,
        ak
    );
    for i in 0..NP {
        text += &format!(
            "  {:<13} {:>9.5} -> {:>9.5} ({:+.5})  g {:+.3e}\n",
            name(i),
            before[i],
            after[i],
            after[i] - before[i],
            ghat[i]
        );
    }
    if !pin.is_empty() {
        text += &format!("  pinned: {}\n", pin.join(", "));
    }
    Ok(IterSummary {
        iter_json,
        candidate,
        text,
    })
}

/// Write evolve-iter-KKK.json, evolve-checkpoint-KKK.json and candidate.json
/// (KKK = iterations completed).
fn write_iter(dir: &Path, st: &EvolveState, s: &IterSummary) -> Result<(), String> {
    write_json(
        dir.join(format!("evolve-iter-{:03}.json", st.iter)),
        &s.iter_json,
    )?;
    write_json(
        dir.join(format!("evolve-checkpoint-{:03}.json", st.iter)),
        &serde_json::to_value(st).unwrap(),
    )?;
    write_json(
        dir.join("candidate.json"),
        &serde_json::to_value(&s.candidate).unwrap(),
    )
}

fn latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(n) = name
            .strip_prefix("evolve-checkpoint-")
            .and_then(|s| s.strip_suffix(".json"))
            .and_then(|s| s.parse::<u64>().ok())
            && best.as_ref().is_none_or(|b| n > b.0)
        {
            best = Some((n, e.path()));
        }
    }
    best.map(|b| b.1)
}

pub fn run(a: EvolveArgs) -> Result<(), String> {
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let parent = load_generation(&a.gen_file)?;
    let openings = load_openings(&a.openings)?;
    if a.shards == 0 {
        return Err("--shards must be >= 1".into());
    }
    let mut st = fresh_state(
        &parent,
        resolve(a.a0, None, DEFAULT_A0)?,
        resolve(a.big_a, None, DEFAULT_BIG_A)?,
        resolve(a.c0, None, DEFAULT_C0)?,
    );
    if a.resume {
        let p = latest_checkpoint(&a.out_dir).ok_or("--resume: no checkpoint in out-dir")?;
        let s = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
        st = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))?;
        st.a0 = resolve(a.a0, Some(st.a0), DEFAULT_A0)?;
        st.big_a = resolve(a.big_a, Some(st.big_a), DEFAULT_BIG_A)?;
        st.c0 = resolve(a.c0, Some(st.c0), DEFAULT_C0)?;
        eprintln!("resumed from {} (iter {})", p.display(), st.iter);
    }
    let t_start = Instant::now();
    while st.iter < a.iters {
        let mut shards = Vec::new();
        for k in 0..a.shards {
            shards.push(play_evolve_shard(
                &st.params, &openings, st.iter, a.seed, k, a.pairs, st.c0, &a.play,
            )?);
        }
        let sum = reduce_iter(&mut st, &shards, &parent.spec)?;
        write_iter(&a.out_dir, &st, &sum)?;
        print!("{}", sum.text);
        if sum.iter_json["finite"] == json!(false) {
            return Err("non-finite update; stopping".into());
        }
        if let Some(m) = a.max_minutes
            && t_start.elapsed().as_secs_f64() / 60.0 >= m
        {
            println!("--max-minutes reached after iteration {}", st.iter);
            break;
        }
    }
    Ok(())
}

pub fn run_shard(a: ShardArgs) -> Result<(), String> {
    let gen_ = load_generation(&a.gen_file)?;
    let openings = load_openings(&a.openings)?;
    let state = load_state(&a.state)?;
    let c0 = match &state {
        Some(s) => {
            if gen_hash(&s.params) != gen_hash(&gen_.constants) {
                return Err(format!(
                    "--state constants (hash {}) differ from --gen constants ({})",
                    gen_hash(&s.params),
                    gen_hash(&gen_.constants)
                ));
            }
            if s.iter != a.iter {
                return Err(format!(
                    "--iter {} but the state expects iteration {}",
                    a.iter, s.iter
                ));
            }
            resolve(a.c0, Some(s.c0), DEFAULT_C0)?
        }
        None => {
            if a.iter != 0 {
                return Err(
                    "--iter > 0 needs --state (the checkpoint of the previous reduce)".into(),
                );
            }
            resolve(a.c0, None, DEFAULT_C0)?
        }
    };
    let mut sh = play_evolve_shard(
        &gen_.constants,
        &openings,
        a.iter,
        a.seed,
        a.shard_index,
        a.pairs,
        c0,
        &a.play,
    )?;
    sh.config["generation"] = json!(gen_.generation);
    sh.config["gen_file"] = json!(a.gen_file);
    sh.config["openings"] = json!(a.openings);
    write_json(a.out.clone(), &serde_json::to_value(&sh).unwrap())?;
    println!(
        "evolve-shard {}: iter {} shard {} | {} pairs in {:.1}s | score+ {:.4} | penta {:?} | c_K {:.4}",
        a.out.display(),
        sh.iter,
        sh.shard_index,
        a.pairs,
        sh.elapsed_ms as f64 / 1000.0,
        sh.score_plus,
        sh.pentanomial,
        sh.c_k
    );
    Ok(())
}

fn read_shards(files: &[PathBuf], dir: &Option<PathBuf>) -> Result<Vec<EvolveShard>, String> {
    let mut files = files.to_vec();
    if let Some(d) = dir {
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
    let shards = read_shards(&a.shards, &a.shard_dir)?;
    let mut st = match load_state(&a.state)? {
        Some(mut s) => {
            s.a0 = resolve(a.a0, Some(s.a0), DEFAULT_A0)?;
            s.big_a = resolve(a.big_a, Some(s.big_a), DEFAULT_BIG_A)?;
            s.c0 = resolve(a.c0, Some(s.c0), DEFAULT_C0)?;
            s
        }
        None => fresh_state(
            &gen_,
            resolve(a.a0, None, DEFAULT_A0)?,
            resolve(a.big_a, None, DEFAULT_BIG_A)?,
            resolve(a.c0, None, DEFAULT_C0)?,
        ),
    };
    if gen_hash(&st.params) != gen_hash(&gen_.constants) {
        return Err(format!(
            "--state constants (hash {}) differ from --gen constants ({})",
            gen_hash(&st.params),
            gen_hash(&gen_.constants)
        ));
    }
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let sum = reduce_iter(&mut st, &shards, &gen_.spec)?;
    write_iter(&a.out_dir, &st, &sum)?;
    print!("{}", sum.text);
    if sum.iter_json["finite"] == json!(false) {
        return Err("non-finite update".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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

    fn data(rel: &str) -> String {
        format!("{}/../../data/{rel}", env!("CARGO_MANIFEST_DIR"))
    }

    fn gen0() -> Generation {
        load_generation(Path::new(&data("generations/gen-000.json"))).unwrap()
    }

    fn argv(v: &[&str]) -> Vec<String> {
        std::iter::once("x")
            .chain(v.iter().copied())
            .map(String::from)
            .collect()
    }

    /// A shard without games: `score_plus = 0.5 + 0.1 * sum_i Delta_i * sign(target_i - theta_i)`.
    fn mock_shard(st: &EvolveState, k: u64, seed: u64, target: &[f32; NP]) -> EvolveShard {
        let delta = derive_delta(seed, st.iter, k);
        let th = theta_of(&st.params);
        let mut score = 0.5;
        for i in 0..NP {
            let s = if target[i] > th[i] {
                1.0
            } else if target[i] < th[i] {
                -1.0
            } else {
                0.0
            };
            score += 0.1 * delta[i] as f64 * s;
        }
        EvolveShard {
            kind: "evolve-shard".into(),
            version: 1,
            spec: SPEC_ID.into(),
            centre_hash: gen_hash(&st.params),
            iter: st.iter,
            seed,
            shard_index: k,
            c_k: c_k(st.c0, st.iter),
            delta: delta.to_vec(),
            score_plus: score,
            pentanomial: [0, 0, 1, 0, 0],
            wdl: Wdl { w: 0, d: 2, l: 0 },
            elapsed_ms: 1,
            config: json!({"depth": 1, "movetime_ms": null, "pairs": 1}),
        }
    }

    #[test]
    fn delta_is_deterministic_and_differs_across_shards() {
        let a = derive_delta(7, 3, 0);
        assert_eq!(a, derive_delta(7, 3, 0));
        assert!(a.iter().all(|&x| x == 1 || x == -1));
        let mut all = std::collections::BTreeSet::new();
        for k in 0..16 {
            all.insert(derive_delta(7, 3, k));
        }
        assert_eq!(all.len(), 16, "16 shards must draw 16 different vectors");
        assert_ne!(derive_delta(7, 3, 0), derive_delta(7, 4, 0));
        assert_ne!(derive_delta(7, 3, 0), derive_delta(8, 3, 0));
        assert_eq!(shard_seed(7, 3, 2), util::mix(util::mix(7, 3), 2));
        // both signs occur
        assert!(a.contains(&1) || derive_delta(7, 3, 1).contains(&1));
    }

    #[test]
    fn schedules_match_their_formulas() {
        assert_eq!(c_k(0.05, 0), 0.05);
        assert_eq!(a_k(0.1, 10.0, 0), 0.1 / 11f64.powf(0.602));
        for k in [1u64, 5, 99] {
            let kk = k as f64;
            assert!((c_k(0.05, k) - 0.05 / (kk + 1.0).powf(0.101)).abs() < 1e-15);
            assert!((a_k(0.1, 10.0, k) - 0.1 / (kk + 11.0).powf(0.602)).abs() < 1e-15);
        }
        // cooling: both decrease with K
        assert!(c_k(0.05, 10) < c_k(0.05, 0) && a_k(0.1, 10.0, 10) < a_k(0.1, 10.0, 0));
        // a_K at K = 0 from the spec numbers
        assert!((a_k(0.1, 10.0, 0) - 0.0236).abs() < 1e-3);
    }

    #[test]
    fn twins_are_relative_and_clamped() {
        let mut p = gen0().constants;
        let d = derive_delta(1, 0, 0);
        let (pl, mi) = twins(&p, &d, 0.05);
        let (t, tp, tm) = (theta_of(&p), theta_of(&pl), theta_of(&mi));
        for i in 0..NP {
            let step = (0.05 * range(i) * d[i] as f64) as f32;
            let (lo, hi) = bounds(i);
            let tol = 1e-5 * range(i) as f32;
            assert!((tp[i] - (t[i] + step).clamp(lo, hi)).abs() < tol, "{i}");
            assert!((tm[i] - (t[i] - step).clamp(lo, hi)).abs() < tol, "{i}");
        }
        assert_eq!(pl.c, p.c);
        // on a bound the twin is clamped, never outside
        p.alpha = 1.0;
        p.f_acc = 0.0;
        let all_plus = [1i8; NP];
        let (pl, mi) = twins(&p, &all_plus, 0.3);
        assert_eq!(pl.alpha, 1.0);
        assert!(mi.alpha < 1.0);
        assert_eq!(mi.f_acc, 0.0);
        assert!(pl.f_acc > 0.0);
        assert!(pl.validate().is_ok() && mi.validate().is_ok());
    }

    #[test]
    fn reduce_moves_theta_toward_the_target_on_a_synthetic_score() {
        let g = gen0();
        let start = theta_of(&g.constants);
        let mut target = start;
        for i in 0..NP {
            // push each parameter 15% of its range, alternating direction (inside bounds)
            let (lo, hi) = bounds(i);
            let up = i % 2 == 0;
            let t = start[i] as f64 + if up { 0.15 } else { -0.15 } * range(i);
            target[i] = (t as f32).clamp(lo, hi);
        }
        let dist = |p: &Params| -> f64 {
            let t = theta_of(p);
            (0..NP)
                .map(|i| (t[i] - target[i]).abs() as f64 / range(i))
                .sum::<f64>()
                / NP as f64
        };
        let mut st = fresh_state(&g, DEFAULT_A0, DEFAULT_BIG_A, DEFAULT_C0);
        let d0 = dist(&st.params);
        assert!(d0 > 0.05);
        for _ in 0..30 {
            let shards: Vec<EvolveShard> =
                (0..8).map(|k| mock_shard(&st, k, 11, &target)).collect();
            let s = reduce_iter(&mut st, &shards, &g.spec).unwrap();
            assert_eq!(s.candidate.parent, Some(g.generation));
            assert_eq!(s.candidate.training["method"], METHOD);
        }
        let d1 = dist(&st.params);
        assert!(d1 < 0.5 * d0, "distance {d0} -> {d1}");
        assert_eq!(st.iter, 30);
        assert_eq!(st.history.len(), 30);
        assert!(st.params.validate().is_ok());
    }

    #[test]
    fn reduce_refuses_mismatched_centre_iter_and_tampering() {
        let g = gen0();
        let target = theta_of(&g.constants);
        let st0 = fresh_state(&g, DEFAULT_A0, DEFAULT_BIG_A, DEFAULT_C0);
        let ok = mock_shard(&st0, 0, 5, &target);
        let try_reduce = |shards: &[EvolveShard]| {
            let mut st = st0.clone();
            reduce_iter(&mut st, shards, &g.spec)
        };
        assert!(try_reduce(&[ok.clone(), mock_shard(&st0, 1, 5, &target)]).is_ok());
        // different centre_hash between shards
        let mut b = mock_shard(&st0, 1, 5, &target);
        b.centre_hash = "0000000000000000".into();
        let e = try_reduce(&[ok.clone(), b.clone()]).unwrap_err();
        assert!(e.contains("centre_hash"), "{e}");
        // all agree with each other but not with the state
        let e = try_reduce(&[b]).unwrap_err();
        assert!(e.contains("centre_hash"), "{e}");
        // different iter between shards
        let mut c = mock_shard(&st0, 1, 5, &target);
        c.iter = 1;
        let e = try_reduce(&[ok.clone(), c.clone()]).unwrap_err();
        assert!(e.contains("iter"), "{e}");
        // iter that the state does not expect
        let e = try_reduce(&[c]).unwrap_err();
        assert!(e.contains("iter"), "{e}");
        // duplicate shard index, tampered delta, wrong c_k
        let e = try_reduce(&[ok.clone(), ok.clone()]).unwrap_err();
        assert!(e.contains("duplicate"), "{e}");
        let mut d = mock_shard(&st0, 1, 5, &target);
        d.delta[0] = -d.delta[0];
        assert!(try_reduce(&[ok.clone(), d]).unwrap_err().contains("delta"));
        let mut e2 = mock_shard(&st0, 1, 5, &target);
        e2.c_k *= 2.0;
        assert!(try_reduce(&[ok, e2]).unwrap_err().contains("c_k"));
        assert!(try_reduce(&[]).is_err());
    }

    #[test]
    fn checkpoint_and_candidate_files_round_trip() {
        let g = gen0();
        let dir = std::env::temp_dir().join(format!("kepler-evolve-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let target = theta_of(&g.constants);
        let mut st = fresh_state(&g, DEFAULT_A0, DEFAULT_BIG_A, DEFAULT_C0);
        let sh: Vec<EvolveShard> = (0..2).map(|k| mock_shard(&st, k, 3, &target)).collect();
        let mut files = Vec::new();
        for (k, s) in sh.iter().enumerate() {
            let f = dir.join(format!("s{k}.json"));
            write_json(f.clone(), &serde_json::to_value(s).unwrap()).unwrap();
            files.push(f.to_str().unwrap().to_string());
        }
        let sum = reduce_iter(&mut st, &sh, &g.spec).unwrap();
        write_iter(&dir, &st, &sum).unwrap();
        for f in [
            "evolve-checkpoint-001.json",
            "evolve-iter-001.json",
            "candidate.json",
        ] {
            assert!(dir.join(f).exists(), "{f}");
        }
        let back: EvolveState = serde_json::from_str(
            &std::fs::read_to_string(dir.join("evolve-checkpoint-001.json")).unwrap(),
        )
        .unwrap();
        assert_eq!((back.iter, back.history.len()), (1, 1));
        assert_eq!(back.big_a, 10.0);
        assert!(
            std::fs::read_to_string(dir.join("evolve-checkpoint-001.json"))
                .unwrap()
                .contains("\"A\"")
        );
        let cand =
            Generation::from_json(&std::fs::read_to_string(dir.join("candidate.json")).unwrap())
                .unwrap();
        assert_eq!(cand.parent, Some(g.generation));
        assert_eq!(cand.training["method"], METHOD);
        assert_eq!(
            latest_checkpoint(&dir).unwrap().file_name().unwrap(),
            "evolve-checkpoint-001.json"
        );
        // the reduce command reads the shard files and the previous candidate/state
        let out2 = dir.join("out2");
        let gp = data("generations/gen-000.json");
        let mut v = vec![
            "--gen",
            &gp,
            "--out-dir",
            out2.to_str().unwrap(),
            "--A",
            "10",
            "--shards",
        ];
        v.extend(files.iter().map(String::as_str));
        run_reduce(WR::try_parse_from(argv(&v)).unwrap().a).unwrap();
        assert!(out2.join("candidate.json").exists());
        let c1: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("candidate.json")).unwrap())
                .unwrap();
        let c2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out2.join("candidate.json")).unwrap())
                .unwrap();
        assert_eq!(c1["constants"], c2["constants"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn tiny_real_shard_runs_and_is_deterministic() {
        // one pair at depth 1, 16 plies: just proves the game plumbing end to end
        let g = gen0();
        let ops = load_openings(Path::new(&data("openings/balanced-200.epd"))).unwrap();
        let play = WS::try_parse_from(argv(&[
            "--gen",
            "x",
            "--openings",
            "x",
            "--out",
            "x",
            "--depth",
            "1",
            "--threads",
            "1",
            "--max-plies",
            "16",
            "--explore-plies",
            "2",
        ]))
        .unwrap()
        .a
        .play;
        let a = play_evolve_shard(&g.constants, &ops, 0, 9, 0, 1, 0.05, &play).unwrap();
        let b = play_evolve_shard(&g.constants, &ops, 0, 9, 0, 1, 0.05, &play).unwrap();
        assert_eq!(a.pentanomial, b.pentanomial);
        assert_eq!(a.score_plus, b.score_plus);
        assert_eq!(a.delta, derive_delta(9, 0, 0).to_vec());
        assert!((0.0..=1.0).contains(&a.score_plus));
        assert_eq!(a.pentanomial.iter().sum::<u32>(), 1);
        assert_eq!(a.centre_hash, gen_hash(&g.constants));
        // and the shard feeds the reducer
        let mut st = fresh_state(&g, DEFAULT_A0, DEFAULT_BIG_A, DEFAULT_C0);
        assert!(reduce_iter(&mut st, &[a], &g.spec).is_ok());
    }
}
