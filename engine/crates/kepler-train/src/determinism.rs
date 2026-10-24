//! Determinism reference units (rule D8, `docs/v1/08_DISTRIBUTED_COMPUTE.md` §3a).
//!
//! `determinism-ref` runs three fixed tiny units and writes their shard files plus a
//! `reference.json` of `{kind, args, result_hash}`. `determinism-check` reruns the units
//! from a `reference.json` and exits non-zero if any `result_hash` differs. The reference
//! is made on the maintainer's laptop and committed; every other machine must match it.
//!
//! Paths in `args` are relative to the engine directory (`--engine-dir`, default `.`).
//! They are not part of `result_hash`, so the directory can be anywhere.

use crate::{evolve, sprt, tdleaf};
use clap::{Args, Parser};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct RefArgs {
    /// Directory for the three shard files and reference.json
    #[arg(long = "out-dir")]
    out_dir: PathBuf,
    /// The engine directory that holds data/ (default: the current directory)
    #[arg(long = "engine-dir", default_value = ".")]
    engine_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct CheckArgs {
    /// reference.json written by determinism-ref
    #[arg(long)]
    reference: PathBuf,
    /// The engine directory that holds data/ (default: the current directory)
    #[arg(long = "engine-dir", default_value = ".")]
    engine_dir: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RefUnit {
    kind: String,
    args: Vec<String>,
    result_hash: String,
}

#[derive(Parser)]
struct TdW {
    #[command(flatten)]
    a: tdleaf::ShardArgs,
}
#[derive(Parser)]
struct SprtW {
    #[command(flatten)]
    a: sprt::ShardArgs,
}
#[derive(Parser)]
struct EvW {
    #[command(flatten)]
    a: evolve::ShardArgs,
}

/// The fixed units: (kind, args without `--out`).
fn units() -> Vec<(&'static str, Vec<String>)> {
    let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    vec![
        (
            "tdleaf-shard",
            v(&[
                "--gen",
                "data/generations/gen-000.json",
                "--openings",
                "data/openings/balanced-200.epd",
                "--games",
                "6",
                "--depth",
                "2",
                "--seed",
                "424242",
                "--shard-index",
                "0",
                "--threads",
                "2",
            ]),
        ),
        (
            "sprt-shard",
            v(&[
                "--a",
                "data/generations/gen-000.json",
                "--b",
                "data/ablations/gen-000-material-only.json",
                "--openings",
                "data/openings/balanced-200.epd",
                "--pairs",
                "3",
                "--depth",
                "2",
                "--seed",
                "424242",
                "--threads",
                "2",
            ]),
        ),
        (
            "evolve-shard",
            v(&[
                "--gen",
                "data/generations/gen-000.json",
                "--state",
                "none",
                "--openings",
                "data/openings/balanced-200.epd",
                "--iter",
                "0",
                "--pairs",
                "2",
                "--depth",
                "2",
                "--seed",
                "424242",
                "--threads",
                "2",
            ]),
        ),
    ]
}

/// Run one shard command in-process and return the `result_hash` of the file it wrote.
pub fn run_unit(kind: &str, args: &[String], out: &Path) -> Result<String, String> {
    let argv: Vec<String> = std::iter::once("kepler-train".to_string())
        .chain(args.iter().cloned())
        .chain(["--out".to_string(), out.display().to_string()])
        .collect();
    let bad = |e: clap::Error| e.to_string();
    match kind {
        "tdleaf-shard" => tdleaf::run_shard(TdW::try_parse_from(&argv).map_err(bad)?.a)?,
        "sprt-shard" => sprt::run_shard(SprtW::try_parse_from(&argv).map_err(bad)?.a)?,
        "evolve-shard" => evolve::run_shard(EvW::try_parse_from(&argv).map_err(bad)?.a)?,
        k => return Err(format!("unknown unit kind {k}")),
    }
    let s = std::fs::read_to_string(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| e.to_string())?;
    v["result_hash"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{}: no result_hash", out.display()))
}

/// Prefix the `data/...` paths of a reference unit with the engine directory.
fn resolve(args: &[String], engine_dir: &Path) -> Vec<String> {
    args.iter()
        .map(|a| {
            if a.starts_with("data/") {
                engine_dir.join(a).display().to_string()
            } else {
                a.clone()
            }
        })
        .collect()
}

pub fn run_ref(a: RefArgs) -> Result<(), String> {
    std::fs::create_dir_all(&a.out_dir).map_err(|e| e.to_string())?;
    let mut list = Vec::new();
    for (kind, args) in units() {
        let out = a.out_dir.join(format!("{kind}.json"));
        let hash = run_unit(kind, &resolve(&args, &a.engine_dir), &out)?;
        println!("{kind}: result_hash {hash}");
        list.push(RefUnit {
            kind: kind.into(),
            args,
            result_hash: hash,
        });
    }
    let doc = json!({
        "note": "reference units for determinism rule D8; produced by `kepler-train determinism-ref`",
        "units": list,
    });
    let p = a.out_dir.join("reference.json");
    std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", p.display()))?;
    println!("wrote {}", p.display());
    Ok(())
}

pub fn run_check(a: CheckArgs) -> Result<(), String> {
    let s = std::fs::read_to_string(&a.reference)
        .map_err(|e| format!("{}: {e}", a.reference.display()))?;
    let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| e.to_string())?;
    let units: Vec<RefUnit> =
        serde_json::from_value(v["units"].clone()).map_err(|e| format!("reference.json: {e}"))?;
    if units.is_empty() {
        return Err("reference.json lists no units".into());
    }
    let tmp = std::env::temp_dir().join(format!("kepler-determinism-check-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let mut bad = 0;
    for u in &units {
        let out = tmp.join(format!("{}.json", u.kind));
        match run_unit(&u.kind, &resolve(&u.args, &a.engine_dir), &out) {
            Ok(h) if h == u.result_hash => println!("OK       {}  {h}", u.kind),
            Ok(h) => {
                bad += 1;
                println!("MISMATCH {}  got {h}, reference {}", u.kind, u.result_hash);
            }
            Err(e) => {
                bad += 1;
                println!("MISMATCH {}  error: {e}", u.kind);
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    if bad > 0 {
        return Err(format!("{bad} determinism check(s) failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn data(rel: &str) -> String {
        format!("{}/../../data/{rel}", env!("CARGO_MANIFEST_DIR"))
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kepler-det-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn read(p: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    /// Unit args (no --out) for each kind, small and at depth 2.
    fn tiny(kind: &str, seed: &str, extra: &[&str]) -> Vec<String> {
        let (g, ops) = (
            data("generations/gen-000.json"),
            data("openings/balanced-200.epd"),
        );
        let mut a = match kind {
            "tdleaf-shard" => s(&[
                "--gen",
                &g,
                "--openings",
                &ops,
                "--games",
                "4",
                "--depth",
                "2",
                "--max-plies",
                "40",
                "--explore-plies",
                "2",
                "--seed",
                seed,
            ]),
            "sprt-shard" => s(&[
                "--a",
                &g,
                "--b",
                &data("ablations/gen-000-material-only.json"),
                "--openings",
                &ops,
                "--pairs",
                "2",
                "--depth",
                "2",
                "--max-plies",
                "40",
                "--seed",
                seed,
            ]),
            _ => s(&[
                "--gen",
                &g,
                "--state",
                "none",
                "--openings",
                &ops,
                "--pairs",
                "2",
                "--depth",
                "2",
                "--max-plies",
                "40",
                "--explore-plies",
                "2",
                "--seed",
                seed,
            ]),
        };
        a.extend(extra.iter().map(|x| x.to_string()));
        a
    }

    fn threads_equal(kind: &str) {
        let d = tmp(&format!("threads-{kind}"));
        let h = |t: &str| {
            let mut a = tiny(kind, "77", &["--threads", t]);
            // the same unit, only --threads differs
            a.extend(s(&["--hash", "4"]));
            run_unit(kind, &a, &d.join(format!("t{t}.json"))).unwrap()
        };
        let (h1, h3) = (h("1"), h("3"));
        assert_eq!(h1, h3, "{kind}: result_hash depends on the thread count");
        // and the files agree on everything but the excluded fields
        let (a, b) = (read(&d.join("t1.json")), read(&d.join("t3.json")));
        assert_eq!(a["config"]["threads"], 1);
        assert_eq!(b["config"]["threads"], 3);
        assert_eq!(a["deterministic"], true);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn tdleaf_shard_threads_1_vs_3() {
        threads_equal("tdleaf-shard");
    }
    #[test]
    fn sprt_shard_threads_1_vs_3() {
        threads_equal("sprt-shard");
    }
    #[test]
    fn evolve_shard_threads_1_vs_3() {
        threads_equal("evolve-shard");
    }

    #[test]
    fn time_limits_are_refused_unless_allowed() {
        let d = tmp("d3");
        for kind in ["tdleaf-shard", "sprt-shard", "evolve-shard"] {
            let mut a = tiny(kind, "5", &["--movetime", "5", "--threads", "1"]);
            if kind != "tdleaf-shard" {
                // sprt / evolve take exactly one of --depth, --movetime
                let i = a.iter().position(|x| x == "--depth").unwrap();
                a.drain(i..i + 2);
            }
            let out = d.join(format!("{kind}.json"));
            let e = run_unit(kind, &a, &out).unwrap_err();
            assert!(e.contains("determinism rule D3"), "{kind}: {e}");
            assert!(!out.exists());
            a.push("--allow-nondeterministic".into());
            run_unit(kind, &a, &out).unwrap();
            assert_eq!(read(&out)["deterministic"], false, "{kind}");
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn reducers_refuse_mixed_determinism_and_edited_shards() {
        let d = tmp("mixed");
        let a = tiny("sprt-shard", "1", &["--threads", "1"]);
        let (f1, f2) = (d.join("a.json"), d.join("b.json"));
        run_unit("sprt-shard", &a, &f1).unwrap();
        let b = tiny(
            "sprt-shard",
            "2",
            &["--threads", "1", "--allow-nondeterministic"],
        );
        run_unit("sprt-shard", &b, &f2).unwrap();
        let load = |p: &Path| -> sprt::SprtShard { crate::util::load_shard(p).unwrap() };
        assert!(sprt::reduce(&[load(&f1), load(&f2)], 0.0, 10.0, 0.05, 0.05).is_err());
        // an edited shard fails the hash check at load time
        let mut v = read(&f1);
        v["pentanomial"][2] = json!(99);
        std::fs::write(&f1, v.to_string()).unwrap();
        let e = crate::util::load_shard::<sprt::SprtShard>(&f1).unwrap_err();
        assert!(e.contains("result_hash"), "{e}");
        let _ = std::fs::remove_dir_all(d);
    }

    /// Remove fields that legitimately differ between runs, at any depth.
    fn strip_volatile(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.retain(|k, _| {
                    ![
                        "created",
                        "elapsed_ms",
                        "seconds",
                        "shard_seconds_sum",
                        "games_per_sec",
                    ]
                    .contains(&k.as_str())
                });
                m.values_mut().for_each(strip_volatile);
            }
            Value::Array(a) => a.iter_mut().for_each(strip_volatile),
            _ => {}
        }
    }

    fn shuffled<T: Clone>(v: &[T]) -> Vec<T> {
        let mut w = v.to_vec();
        w.reverse();
        w.rotate_left(1);
        w
    }

    #[test]
    fn tdleaf_reduce_is_order_independent() {
        let d = tmp("red-td");
        let g = data("generations/gen-000.json");
        let mut files = Vec::new();
        for k in 0..3 {
            let mut a = tiny(
                "tdleaf-shard",
                "9",
                &["--threads", "2", "--shard-index", &k.to_string()],
            );
            let a2 = a.split_off(0);
            let f = d.join(format!("s{k}.json"));
            run_unit("tdleaf-shard", &a2, &f).unwrap();
            files.push(f.display().to_string());
        }
        let reduce = |fs: &[String], out: &str| {
            let mut argv = s(&[
                "x",
                "--gen",
                &g,
                "--state",
                "none",
                "--out-dir",
                out,
                "--shards",
            ]);
            argv.extend(fs.iter().cloned());
            #[derive(Parser)]
            struct W {
                #[command(flatten)]
                a: tdleaf::ReduceArgs,
            }
            tdleaf::run_reduce(W::try_parse_from(argv).unwrap().a).unwrap();
            let mut round = read(&Path::new(out).join("round-001.json"));
            let mut cand = read(&Path::new(out).join("candidate.json"));
            let mut ck = read(&Path::new(out).join("checkpoint-001.json"));
            for v in [&mut round, &mut cand, &mut ck] {
                strip_volatile(v);
            }
            (round, cand, ck)
        };
        let o1 = d.join("o1");
        let o2 = d.join("o2");
        let r1 = reduce(&files, o1.to_str().unwrap());
        let r2 = reduce(&shuffled(&files), o2.to_str().unwrap());
        assert_eq!(r1.0, r2.0);
        assert_eq!(r1.1, r2.1);
        assert_eq!(r1.2, r2.2);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn sprt_reduce_is_order_independent() {
        let d = tmp("red-sprt");
        let mut files = Vec::new();
        for k in 0..3 {
            let a = tiny(
                "sprt-shard",
                "9",
                &["--threads", "2", "--shard-index", &k.to_string()],
            );
            let f = d.join(format!("s{k}.json"));
            run_unit("sprt-shard", &a, &f).unwrap();
            files.push(f);
        }
        let reduce = |fs: &[PathBuf], out: &str| {
            let a = sprt::ReduceArgs::for_test(fs.to_vec(), PathBuf::from(out));
            sprt::run_reduce(a).unwrap();
            let mut v = read(Path::new(out));
            strip_volatile(&mut v);
            v
        };
        let (o1, o2) = (d.join("o1.json"), d.join("o2.json"));
        let r1 = reduce(&files, o1.to_str().unwrap());
        let r2 = reduce(&shuffled(&files), o2.to_str().unwrap());
        assert_eq!(r1, r2);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn evolve_reduce_is_order_independent() {
        let d = tmp("red-ev");
        let g = data("generations/gen-000.json");
        let mut files = Vec::new();
        for k in 0..3 {
            let a = tiny(
                "evolve-shard",
                "9",
                &["--threads", "2", "--shard-index", &k.to_string()],
            );
            let f = d.join(format!("s{k}.json"));
            run_unit("evolve-shard", &a, &f).unwrap();
            files.push(f.display().to_string());
        }
        let reduce = |fs: &[String], out: &str| {
            let mut argv = s(&[
                "x",
                "--gen",
                &g,
                "--state",
                "none",
                "--out-dir",
                out,
                "--shards",
            ]);
            argv.extend(fs.iter().cloned());
            #[derive(Parser)]
            struct W {
                #[command(flatten)]
                a: evolve::ReduceArgs,
            }
            evolve::run_reduce(W::try_parse_from(argv).unwrap().a).unwrap();
            let mut out_v = Vec::new();
            for f in [
                "evolve-checkpoint-001.json",
                "evolve-iter-001.json",
                "candidate.json",
            ] {
                let mut v = read(&Path::new(out).join(f));
                strip_volatile(&mut v);
                out_v.push(v);
            }
            out_v
        };
        let (o1, o2) = (d.join("o1"), d.join("o2"));
        let r1 = reduce(&files, o1.to_str().unwrap());
        let r2 = reduce(&shuffled(&files), o2.to_str().unwrap());
        assert_eq!(r1, r2);
        let _ = std::fs::remove_dir_all(d);
    }
}
