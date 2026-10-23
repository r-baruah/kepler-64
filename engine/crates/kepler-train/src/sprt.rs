//! `kepler-train sprt`: SPRT between two generation files by in-process self-play.
//!
//! Games are played in pairs (same opening, colours swapped). Pairs are scored in pair
//! order (finished pairs wait for earlier ones) so stopping early cannot bias the sample.
//! Statistics: `stats::llr_penta` (pentanomial GSPRT, normal approximation).
//!
//! The engines are deterministic, so a pair would repeat whenever an opening comes
//! round again, and identical engines would score exactly 1.0 on every pair (zero
//! variance). Both games of a pair therefore use random exploration for `--explore-plies`
//! (default 6) plies after the opening, seeded per game, as in `selfplay`.

#![allow(clippy::needless_range_loop)]

use crate::chess::End;
use crate::selfgame::{
    self, AdjArgs, GameCfg, Opening, gen_hash, load_generation, load_openings, par_run,
};
use crate::stats;
use crate::util::{self, XorShift};
use clap::Args;
use kepler_core::Params;
use kepler_core::params::SPEC_ID;
use kepler_core::search::{Limits, Searcher};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Args, Debug)]
pub struct SprtArgs {
    #[arg(long)]
    a: PathBuf,
    #[arg(long)]
    b: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long)]
    depth: Option<u32>,
    /// Milliseconds per move
    #[arg(long)]
    movetime: Option<u64>,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long = "max-games", default_value_t = 4000)]
    max_games: usize,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    elo0: f64,
    #[arg(long, default_value_t = 10.0, allow_hyphen_values = true)]
    elo1: f64,
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
    #[arg(long, default_value_t = 0.05)]
    beta: f64,
    /// Do not stop on a bound before this many games (the normal approximation is
    /// unreliable on a handful of pairs)
    #[arg(long = "min-games", default_value_t = 100)]
    min_games: usize,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 6)]
    explore_plies: usize,
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[command(flatten)]
    adj: AdjArgs,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Default)]
struct Shared {
    next: usize,
    done: BTreeMap<usize, Result<(f64, [u32; 3]), String>>,
    evaluated: usize,
    penta: [u32; 5],
    wdl: [u32; 3],
    llr: f64,
    stop: Option<&'static str>,
    err: Option<String>,
}

/// Points for engine A (white `a_white`) from the result.
fn points(e: End, a_white: bool) -> f64 {
    match (e, a_white) {
        (End::Draw, _) => 0.5,
        (End::WhiteWins, true) | (End::BlackWins, false) => 1.0,
        _ => 0.0,
    }
}

/// Play pair `k` (A white, then A black, same opening). Returns A's points (0..=2) and
/// the W/D/L counts for A. Game seeds are `mix(seed, 2k + a_white)`: with a shared seed,
/// identical engines would replay the same game with colours swapped and every pair
/// would tie 1.0.
pub(crate) fn play_pair(
    cfg: &GameCfg,
    o: &Opening,
    seed: u64,
    k: usize,
    pa: &Params,
    pb: &Params,
    st: &mut [Searcher; 2],
) -> Result<(f64, [u32; 3]), String> {
    let mut pts = 0.0;
    let mut wdl = [0u32; 3];
    for a_white in [true, false] {
        let gseed = util::mix(seed, 2 * k as u64 + a_white as u64);
        st[0].clear();
        st[1].clear();
        let [sa, sb] = st;
        let g = if a_white {
            selfgame::play_game(cfg, o, [sa, sb], [pa, pb], gseed)?
        } else {
            selfgame::play_game(cfg, o, [sb, sa], [pb, pa], gseed)?
        };
        let p = points(g.result, a_white);
        pts += p;
        wdl[if p == 1.0 {
            0
        } else if p == 0.5 {
            1
        } else {
            2
        }] += 1;
    }
    Ok((pts, wdl))
}

pub(crate) fn limits_from(depth: Option<u32>, movetime: Option<u64>) -> Result<Limits, String> {
    match (depth, movetime) {
        (Some(d), None) => Ok(Limits {
            depth: Some(d),
            ..Default::default()
        }),
        (None, Some(m)) => Ok(Limits {
            movetime_ms: Some(m),
            ..Default::default()
        }),
        _ => Err("give exactly one of --depth, --movetime".into()),
    }
}

#[derive(Args, Debug)]
pub struct ShardArgs {
    #[arg(long)]
    a: PathBuf,
    #[arg(long)]
    b: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    /// Opening pairs (2 games each) in this shard
    #[arg(long, default_value_t = 50)]
    pairs: usize,
    /// Shard seed: it shuffles the opening order and seeds every game. Give each shard of
    /// a test its own seed; with --shard-index the shard seed is mix(seed, shard-index)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long = "shard-index")]
    shard_index: Option<u64>,
    #[arg(long)]
    depth: Option<u32>,
    /// Milliseconds per move
    #[arg(long)]
    movetime: Option<u64>,
    #[arg(long, default_value_t = 10)]
    threads: usize,
    #[arg(long = "max-plies", default_value_t = 300)]
    max_plies: usize,
    #[arg(long = "explore-plies", default_value_t = 6)]
    explore_plies: usize,
    #[arg(long, default_value_t = 8)]
    hash: usize,
    #[command(flatten)]
    adj: AdjArgs,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wdl {
    pub w: u32,
    pub d: u32,
    pub l: u32,
}

/// One SPRT work unit (JSON file written by `sprt-shard`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SprtShard {
    pub kind: String,
    pub version: u32,
    pub spec: String,
    pub a_hash: String,
    pub b_hash: String,
    pub seed: u64,
    pub pentanomial: [u32; 5],
    pub wdl: Wdl,
    pub elapsed_ms: u64,
    pub config: serde_json::Value,
}

pub fn run_shard(a: ShardArgs) -> Result<(), String> {
    let ga = load_generation(&a.a)?;
    let gb = load_generation(&a.b)?;
    let (pa, pb) = (ga.constants, gb.constants);
    let openings = load_openings(&a.openings)?;
    let limits = limits_from(a.depth, a.movetime)?;
    let seed = match a.shard_index {
        Some(k) => util::mix(a.seed, k),
        None => a.seed,
    };
    let cfg = GameCfg {
        limits,
        explore_depth: a.depth.map_or(3, |d| d.saturating_sub(1).max(1)),
        max_plies: a.max_plies,
        explore_plies: a.explore_plies,
        explore_margin: 0.15,
        record: false,
        quiet_leaf: false,
        temperature: 4.0,
        adj: a.adj.cfg(),
    };
    let mut order: Vec<usize> = (0..openings.len()).collect();
    let mut rng = XorShift::new(seed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let t0 = Instant::now();
    let pairs = par_run(
        a.pairs,
        a.threads,
        || [Searcher::new(a.hash), Searcher::new(a.hash)],
        |st, k| {
            let o = &openings[order[k % order.len()]];
            play_pair(&cfg, o, seed, k, &pa, &pb, st)
        },
    );
    let mut penta = [0u32; 5];
    let mut wdl = [0u32; 3];
    for p in pairs {
        let (pts, w) = p?;
        penta[((pts * 2.0).round() as usize).min(4)] += 1;
        for i in 0..3 {
            wdl[i] += w[i];
        }
    }
    let ms = t0.elapsed().as_millis() as u64;
    let mut config = json!({
        "a": { "file": a.a, "generation": ga.generation },
        "b": { "file": a.b, "generation": gb.generation },
        "depth": a.depth, "movetime_ms": a.movetime, "pairs": a.pairs,
        "max_plies": a.max_plies, "explore_plies": a.explore_plies,
        "adjudication": a.adj.cfg(), "openings": a.openings,
        "created": util::utc_iso(),
        "note": "pentanomial = pairs scoring 0, 0.5, 1, 1.5, 2 for A",
    });
    if let Some(k) = a.shard_index {
        config["round_seed"] = json!(a.seed);
        config["shard_index"] = json!(k);
    }
    let sh = SprtShard {
        kind: "sprt-shard".into(),
        version: 1,
        spec: SPEC_ID.into(),
        a_hash: gen_hash(&pa),
        b_hash: gen_hash(&pb),
        seed,
        pentanomial: penta,
        wdl: Wdl {
            w: wdl[0],
            d: wdl[1],
            l: wdl[2],
        },
        elapsed_ms: ms,
        config,
    };
    std::fs::write(&a.out, serde_json::to_string_pretty(&sh).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", a.out.display()))?;
    println!(
        "sprt-shard {}: {} pairs in {:.1}s | penta {:?} | W{} D{} L{}",
        a.out.display(),
        a.pairs,
        ms as f64 / 1000.0,
        penta,
        wdl[0],
        wdl[1],
        wdl[2]
    );
    Ok(())
}

#[derive(Args, Debug)]
pub struct ReduceArgs {
    #[arg(long, num_args = 1.., required = true)]
    shards: Vec<PathBuf>,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    elo0: f64,
    #[arg(long, default_value_t = 10.0, allow_hyphen_values = true)]
    elo1: f64,
    #[arg(long, default_value_t = 0.05)]
    alpha: f64,
    #[arg(long, default_value_t = 0.05)]
    beta: f64,
    #[arg(long)]
    out: PathBuf,
}

/// Combine shards: sum the pentanomials and judge once on the total.
pub fn reduce(
    shards: &[SprtShard],
    elo0: f64,
    elo1: f64,
    alpha: f64,
    beta: f64,
) -> Result<serde_json::Value, String> {
    let first = shards.first().ok_or("no shards")?;
    let mut penta = [0u32; 5];
    let mut wdl = [0u32; 3];
    let mut ms = 0u64;
    for (i, s) in shards.iter().enumerate() {
        if s.kind != "sprt-shard" || s.version != 1 {
            return Err(format!("shard {i}: not an sprt-shard v1 file"));
        }
        if s.spec != first.spec
            || s.a_hash != first.a_hash
            || s.b_hash != first.b_hash
            || s.config.get("depth") != first.config.get("depth")
            || s.config.get("movetime_ms") != first.config.get("movetime_ms")
        {
            return Err(format!(
                "shard {i}: spec / a_hash / b_hash / depth / movetime differ from shard 0"
            ));
        }
        for j in 0..5 {
            penta[j] += s.pentanomial[j];
        }
        wdl[0] += s.wdl.w;
        wdl[1] += s.wdl.d;
        wdl[2] += s.wdl.l;
        ms += s.elapsed_ms;
    }
    let upper = ((1.0 - beta) / alpha).ln();
    let lower = (beta / (1.0 - alpha)).ln();
    let llr = stats::llr_penta(&penta, elo0, elo1);
    let result = if llr >= upper {
        "H1"
    } else if llr <= lower {
        "H0"
    } else {
        "inconclusive"
    };
    let games = penta.iter().sum::<u32>() * 2;
    let (elo, lo, hi) = stats::elo_penta(&penta);
    Ok(json!({
        "llr": llr,
        "bounds": { "lower": lower, "upper": upper },
        "result": result,
        "pentanomial": penta,
        "wdl": { "w": wdl[0], "d": wdl[1], "l": wdl[2] },
        "elo": { "estimate": elo, "lo95": lo, "hi95": hi, "reliable": games >= 200 },
        "games": games,
        "shard_seconds_sum": ms as f64 / 1000.0,
        "config": {
            "a_hash": first.a_hash, "b_hash": first.b_hash, "spec": first.spec,
            "shards": shards.len(), "shard_seeds": shards.iter().map(|s| s.seed).collect::<Vec<_>>(),
            "elo0": elo0, "elo1": elo1, "alpha": alpha, "beta": beta,
            "shard_config": first.config,
            "created": util::utc_iso(),
            "note": "result is for A versus B; judged once on the summed pentanomial",
        },
    }))
}

pub fn run_reduce(a: ReduceArgs) -> Result<(), String> {
    let shards: Vec<SprtShard> = a
        .shards
        .iter()
        .map(|p| {
            let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
        })
        .collect::<Result<_, String>>()?;
    let doc = reduce(&shards, a.elo0, a.elo1, a.alpha, a.beta)?;
    std::fs::write(&a.out, serde_json::to_string_pretty(&doc).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", a.out.display()))?;
    println!(
        "SPRT {}: llr {:.3} in [{:.3}, {:.3}], {} games, penta {}, Elo {:+.1} [{:+.1}, {:+.1}]",
        doc["result"].as_str().unwrap(),
        doc["llr"].as_f64().unwrap(),
        doc["bounds"]["lower"].as_f64().unwrap(),
        doc["bounds"]["upper"].as_f64().unwrap(),
        doc["games"],
        doc["pentanomial"],
        doc["elo"]["estimate"].as_f64().unwrap(),
        doc["elo"]["lo95"].as_f64().unwrap(),
        doc["elo"]["hi95"].as_f64().unwrap(),
    );
    Ok(())
}

pub fn run(a: SprtArgs) -> Result<(), String> {
    let ga = load_generation(&a.a)?;
    let gb = load_generation(&a.b)?;
    let (pa, pb) = (ga.constants, gb.constants);
    let openings = load_openings(&a.openings)?;
    let limits = limits_from(a.depth, a.movetime)?;
    let cfg = GameCfg {
        limits,
        explore_depth: a.depth.map_or(3, |d| d.saturating_sub(1).max(1)),
        max_plies: a.max_plies,
        explore_plies: a.explore_plies,
        explore_margin: 0.15,
        record: false,
        quiet_leaf: false,
        temperature: 4.0,
        adj: a.adj.cfg(),
    };
    let upper = ((1.0 - a.beta) / a.alpha).ln();
    let lower = (a.beta / (1.0 - a.alpha)).ln();
    let max_pairs = a.max_games / 2;
    // shuffled opening order, repeated as needed
    let mut order: Vec<usize> = (0..openings.len()).collect();
    let mut rng = XorShift::new(a.seed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let shared = Mutex::new(Shared::default());
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..a.threads.clamp(1, max_pairs.max(1)) {
            sc.spawn(|| {
                let mut st = [Searcher::new(a.hash), Searcher::new(a.hash)];
                loop {
                    let k = {
                        let mut s = shared.lock().unwrap();
                        if s.stop.is_some() || s.next >= max_pairs {
                            break;
                        }
                        s.next += 1;
                        s.next - 1
                    };
                    let o = &openings[order[k % order.len()]];
                    let res = play_pair(&cfg, o, a.seed, k, &pa, &pb, &mut st);
                    let mut s = shared.lock().unwrap();
                    s.done.insert(k, res);
                    while s.stop.is_none() {
                        let ev = s.evaluated;
                        let Some(r) = s.done.remove(&ev) else { break };
                        s.evaluated += 1;
                        match r {
                            Err(e) => {
                                s.err = Some(e);
                                s.stop = Some("error");
                            }
                            Ok((pts, wdl)) => {
                                s.penta[((pts * 2.0).round() as usize).min(4)] += 1;
                                for i in 0..3 {
                                    s.wdl[i] += wdl[i];
                                }
                                s.llr = stats::llr_penta(&s.penta, a.elo0, a.elo1);
                                if s.evaluated * 2 < a.min_games {
                                } else if s.llr >= upper {
                                    s.stop = Some("H1");
                                } else if s.llr <= lower {
                                    s.stop = Some("H0");
                                }
                                if s.evaluated % 20 == 0 {
                                    eprintln!(
                                        "{} games, llr {:.3} [{:.3}, {:.3}], {:.1}s",
                                        s.evaluated * 2,
                                        s.llr,
                                        lower,
                                        upper,
                                        t0.elapsed().as_secs_f64()
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    let s = shared.into_inner().unwrap();
    if let Some(e) = s.err {
        return Err(e);
    }
    let result = s.stop.unwrap_or("inconclusive");
    let games = s.evaluated * 2;
    let (elo, lo, hi) = stats::elo_penta(&s.penta);
    let secs = t0.elapsed().as_secs_f64();
    let doc = json!({
        "llr": s.llr,
        "bounds": { "lower": lower, "upper": upper },
        "result": result,
        "pentanomial": s.penta,
        "wdl": { "w": s.wdl[0], "d": s.wdl[1], "l": s.wdl[2] },
        "elo": { "estimate": elo, "lo95": lo, "hi95": hi, "reliable": games >= 200 },
        "games": games,
        "seconds": secs,
        "config": {
            "a": { "file": a.a, "generation": ga.generation },
            "b": { "file": a.b, "generation": gb.generation },
            "depth": a.depth, "movetime_ms": a.movetime,
            "elo0": a.elo0, "elo1": a.elo1, "alpha": a.alpha, "beta": a.beta,
            "max_games": a.max_games, "min_games": a.min_games, "max_plies": a.max_plies,
            "explore_plies": a.explore_plies, "openings": a.openings,
            "seed": a.seed, "created": util::utc_iso(),
            "note": "result is for A versus B; pentanomial = pairs scoring 0, 0.5, 1, 1.5, 2 for A",
        },
    });
    std::fs::write(&a.out, serde_json::to_string_pretty(&doc).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", a.out.display()))?;
    println!(
        "SPRT {result}: llr {:.3} in [{:.3}, {:.3}], {games} games (W{} D{} L{}), Elo {:+.1} [{:+.1}, {:+.1}], {:.1}s",
        s.llr, lower, upper, s.wdl[0], s.wdl[1], s.wdl[2], elo, lo, hi, secs
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard(penta: [u32; 5], seed: u64) -> SprtShard {
        SprtShard {
            kind: "sprt-shard".into(),
            version: 1,
            spec: SPEC_ID.into(),
            a_hash: "aa".into(),
            b_hash: "bb".into(),
            seed,
            pentanomial: penta,
            wdl: Wdl { w: 1, d: 2, l: 3 },
            elapsed_ms: 10,
            config: json!({ "depth": 3, "movetime_ms": null }),
        }
    }

    #[test]
    fn reduce_equals_summed_pentanomial_math() {
        let (p1, p2) = ([1, 4, 20, 9, 2], [0, 6, 18, 11, 3]);
        let doc = reduce(&[shard(p1, 1), shard(p2, 2)], 0.0, 10.0, 0.05, 0.05).unwrap();
        let sum = [1, 10, 38, 20, 5];
        assert_eq!(doc["pentanomial"], json!(sum));
        assert_eq!(doc["games"], 148);
        assert_eq!(
            doc["llr"].as_f64().unwrap(),
            stats::llr_penta(&sum, 0.0, 10.0)
        );
        let (e, lo, hi) = stats::elo_penta(&sum);
        assert_eq!(doc["elo"]["estimate"].as_f64().unwrap(), e);
        assert_eq!(doc["elo"]["lo95"].as_f64().unwrap(), lo);
        assert_eq!(doc["elo"]["hi95"].as_f64().unwrap(), hi);
        assert_eq!(doc["wdl"]["w"], 2);
        // judged once on the total: a shard that alone would cross a bound does not decide
        let strong = [0, 0, 5, 20, 40];
        let total = reduce(&[shard(strong, 1)], 0.0, 10.0, 0.05, 0.05).unwrap();
        assert_eq!(total["result"], "H1");
        // mismatched engines are refused
        let mut other = shard(p2, 2);
        other.b_hash = "cc".into();
        assert!(reduce(&[shard(p1, 1), other], 0.0, 10.0, 0.05, 0.05).is_err());
    }
}
