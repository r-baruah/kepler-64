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

#![allow(clippy::needless_range_loop)]

use crate::selfgame::{self, GameCfg, Leaf, load_generation, load_openings, par_run};
use crate::util::{self, XorShift};
use clap::Args;
use kepler_core::params::Generation;
use kepler_core::search::{Limits, Searcher};
use kepler_core::{NGRAD, Params};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    #[arg(long = "games-per-round", default_value_t = 2000)]
    games_per_round: usize,
    #[arg(long, default_value_t = 4)]
    depth: u32,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    /// Adam learning rate in units of each constant's bound range
    #[arg(long, default_value_t = 0.01)]
    lr: f32,
    #[arg(long, default_value_t = 0.7)]
    lambda: f32,
    /// The s in tanh(eval / s), ku
    #[arg(long, default_value_t = 4.0)]
    scale: f32,
    #[arg(long = "max-minutes")]
    max_minutes: Option<f64>,
    #[arg(long)]
    resume: bool,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 8)]
    explore_plies: usize,
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
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
    pub grads: Vec<[f32; NGRAD]>,
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
        grads: leaves.iter().map(|l| l.grad).collect(),
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

fn constants_json(p: &Params) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    for (n, v) in Params::GRAD_NAMES.iter().zip(p.grad_values()) {
        m.insert(n.to_string(), json!(v));
    }
    m.insert("c".into(), json!(p.c));
    m.insert("f_acc".into(), json!(p.f_acc));
    m.into()
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    round: usize,
    games: usize,
    parent: u32,
    params: Params,
    adam: Adam,
    rng_state: u64,
    bound_history: Vec<Vec<String>>,
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

pub fn run(a: TdleafArgs) -> Result<(), String> {
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let parent = load_generation(&a.gen_file)?;
    let openings = load_openings(&a.openings)?;
    let mut ck = Checkpoint {
        round: 0,
        games: 0,
        parent: parent.generation,
        params: parent.constants,
        adam: Adam::new(),
        rng_state: XorShift::new(a.seed).state(),
        bound_history: Vec::new(),
    };
    if a.resume {
        let p = latest_checkpoint(&a.out_dir).ok_or("--resume: no checkpoint in out-dir")?;
        let s = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
        ck = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))?;
        eprintln!("resumed from {} (round {})", p.display(), ck.round);
    }
    let mut rng = XorShift::from_state(ck.rng_state);
    let cfg = GameCfg {
        limits: Limits {
            depth: Some(a.depth),
            ..Default::default()
        },
        explore_depth: a.depth.saturating_sub(1).max(1),
        max_plies: a.max_plies,
        explore_plies: a.explore_plies,
        explore_margin: 0.15,
        record: true,
        scale: a.scale,
    };
    let t_start = Instant::now();
    while ck.round < a.rounds {
        let params = ck.params;
        let round_seed = rng.next_u64();
        let off = rng.below(openings.len());
        let t0 = Instant::now();
        let games = par_run(
            a.games_per_round,
            a.threads,
            || [Searcher::new(a.hash), Searcher::new(a.hash)],
            |st, i| {
                let o = &openings[(off + i / 2) % openings.len()];
                st[0].clear();
                st[1].clear();
                let [s0, s1] = st;
                let g = selfgame::play_game(
                    &cfg,
                    o,
                    [s0, s1],
                    [&params, &params],
                    util::mix(round_seed, i as u64),
                )?;
                if g.moves.is_empty() {
                    return Ok(None);
                }
                let last = g.moves.len() - 1;
                let td = game_td(&g.leaves, g.result_for_mover(last), a.lambda);
                Ok::<_, String>(Some((td, g.result, g.capped)))
            },
        );
        let secs = t0.elapsed().as_secs_f64();
        let mut grad = [0.0f64; NGRAD];
        let (mut plies, mut sum_abs_d, mut sum_d) = (0usize, 0.0f64, 0.0f64);
        let (mut w, mut d, mut l, mut capped, mut n_games) = (0, 0, 0, 0, 0);
        let mut tds = Vec::new();
        for g in games {
            let Some((td, res, cap)) = g? else { continue };
            n_games += 1;
            match res {
                crate::chess::End::WhiteWins => w += 1,
                crate::chess::End::Draw => d += 1,
                crate::chess::End::BlackWins => l += 1,
            }
            capped += cap as usize;
            plies += td.plies;
            sum_abs_d += td.sum_abs_d;
            sum_d += td.sum_d;
            for i in 0..NGRAD {
                grad[i] += td.grad_sum[i] as f64;
            }
            tds.push(td);
        }
        let np = plies.max(1) as f64;
        let mut g32 = [0.0f32; NGRAD];
        for i in 0..NGRAD {
            g32[i] = (grad[i] / np) as f32;
        }
        let before = ck.params.grad_values();
        let proposed = ck.adam.step(before, g32, a.lr);
        let mut next = ck.params;
        next.set_grad_values(proposed);
        let after = next.grad_values();
        let delta: Vec<f32> = (0..NGRAD).map(|i| after[i] - before[i]).collect();
        // first-order change in v this update causes: mean over plies of |grad v . delta|
        let mut pred = 0.0f64;
        for t in &tds {
            for gv in &t.grads {
                pred += (0..NGRAD)
                    .map(|i| (gv[i] * delta[i]) as f64)
                    .sum::<f64>()
                    .abs();
            }
        }
        ck.params = next;
        ck.round += 1;
        ck.games += n_games;
        let bound = at_bound(&ck.params);
        ck.bound_history.push(bound.clone());
        let pin = pinned(&ck.bound_history);
        ck.rng_state = rng.state();
        let finite = g32.iter().all(|x| x.is_finite()) && after.iter().all(|x| x.is_finite());
        let mut names_delta = serde_json::Map::new();
        let mut names_grad = serde_json::Map::new();
        for i in 0..NGRAD {
            names_delta.insert(Params::GRAD_NAMES[i].into(), json!(delta[i]));
            names_grad.insert(Params::GRAD_NAMES[i].into(), json!(g32[i]));
        }
        let rj = json!({
            "round": ck.round, "games": n_games, "plies": plies,
            "wdl": { "white_wins": w, "draws": d, "black_wins": l, "capped": capped },
            "mean_abs_d": sum_abs_d / np, "mean_d": sum_d / np,
            "mean_abs_dv_from_update": pred / np,
            "games_per_sec": n_games as f64 / secs, "seconds": secs,
            "grad_avg": names_grad, "delta": names_delta,
            "constants": constants_json(&ck.params),
            "at_bound": bound, "pinned": pin,
            "depth": a.depth, "lambda": a.lambda, "scale": a.scale, "lr": a.lr,
            "finite": finite,
        });
        let wr = |name: String, v: &serde_json::Value| {
            std::fs::write(
                a.out_dir.join(name),
                serde_json::to_string_pretty(v).unwrap() + "\n",
            )
            .map_err(|e| e.to_string())
        };
        wr(format!("round-{:03}.json", ck.round), &rj)?;
        wr(
            format!("checkpoint-{:03}.json", ck.round),
            &serde_json::to_value(&ck).unwrap(),
        )?;
        let cand = Generation {
            generation: parent.generation + 1,
            parent: Some(ck.parent),
            spec: parent.spec.clone(),
            created: util::utc_date(),
            constants: ck.params,
            pinned: pin.clone(),
            training: json!({
                "rounds": ck.round, "games": ck.games, "depth": a.depth,
                "lambda": a.lambda, "scale": a.scale, "lr": a.lr,
                "lr_units": "relative to bound range (hi - lo)", "optimizer": "adam",
                "games_per_round": a.games_per_round, "seed": a.seed,
            }),
            acceptance: serde_json::Value::Null,
        };
        wr(
            "candidate.json".into(),
            &serde_json::to_value(&cand).unwrap(),
        )?;
        println!(
            "round {} | {} games ({:.2}/s, {:.0}s) | W{} D{} L{} cap{} | mean|d| {:.4} mean d {:+.4} | finite {}",
            ck.round,
            n_games,
            n_games as f64 / secs,
            secs,
            w,
            d,
            l,
            capped,
            sum_abs_d / np,
            sum_d / np,
            finite
        );
        for i in 0..NGRAD {
            println!(
                "  {:<13} {:>9.5} -> {:>9.5} ({:+.5})  grad {:+.3e}",
                Params::GRAD_NAMES[i],
                before[i],
                after[i],
                delta[i],
                g32[i]
            );
        }
        if !pin.is_empty() {
            println!("  pinned: {}", pin.join(", "));
        }
        if !finite {
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

#[cfg(test)]
mod tests {
    use super::*;

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
