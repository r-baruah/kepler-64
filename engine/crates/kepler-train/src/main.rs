//! kepler-train: testing and training harness for Kepler-64 v1.

mod chess;
mod determinism;
mod epd;
mod evolve;
mod matchcmd;
mod openings;
mod selfgame;
mod selfplaycmd;
mod sprt;
mod stats;
mod suite;
mod tdleaf;
mod uci;
mod util;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kepler-train", version, about = "Kepler-64 v1 testing harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Play games between two UCI engines
    Match(matchcmd::MatchArgs),
    /// Self-play games of one generation as JSONL
    Selfplay(selfplaycmd::SelfplayArgs),
    /// TDLeaf(lambda) training rounds
    Tdleaf(tdleaf::TdleafArgs),
    /// One work unit of a TDLeaf round: play games, write summed gradients
    TdleafShard(tdleaf::ShardArgs),
    /// Sum TDLeaf shards and take one Adam step
    TdleafReduce(tdleaf::ReduceArgs),
    /// Cosmological natural selection (SPSA): twin universes play, constants follow the winner
    Evolve(evolve::EvolveArgs),
    /// One work unit of an evolve iteration: theta+ vs theta- mini-match
    EvolveShard(evolve::ShardArgs),
    /// Average evolve shards' SPSA gradients and take one step
    EvolveReduce(evolve::ReduceArgs),
    /// SPRT between two generation files (in-process self-play)
    Sprt(sprt::SprtArgs),
    /// One fixed-size batch of SPRT pairs
    SprtShard(sprt::ShardArgs),
    /// Combine SPRT shards and judge once on the total
    SprtReduce(sprt::ReduceArgs),
    /// Run the fixed reference units and write reference.json (determinism rule D8)
    DeterminismRef(determinism::RefArgs),
    /// Rerun the reference units and compare result hashes; non-zero exit on a mismatch
    DeterminismCheck(determinism::CheckArgs),
    /// Run an EPD suite through one UCI engine
    Suite(suite::SuiteArgs),
    /// Opening suite tools
    Openings {
        #[command(subcommand)]
        cmd: OpeningsCmd,
    },
}

#[derive(Subcommand)]
enum OpeningsCmd {
    /// Build the balanced opening suite with Stockfish
    Build(openings::BuildArgs),
}

fn main() {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Match(a) => matchcmd::run(a),
        Cmd::Selfplay(a) => selfplaycmd::run(a),
        Cmd::Tdleaf(a) => tdleaf::run(a),
        Cmd::TdleafShard(a) => tdleaf::run_shard(a),
        Cmd::TdleafReduce(a) => tdleaf::run_reduce(a),
        Cmd::Evolve(a) => evolve::run(a),
        Cmd::EvolveShard(a) => evolve::run_shard(a),
        Cmd::EvolveReduce(a) => evolve::run_reduce(a),
        Cmd::Sprt(a) => sprt::run(a),
        Cmd::SprtShard(a) => sprt::run_shard(a),
        Cmd::SprtReduce(a) => sprt::run_reduce(a),
        Cmd::DeterminismRef(a) => determinism::run_ref(a),
        Cmd::DeterminismCheck(a) => determinism::run_check(a),
        Cmd::Suite(a) => suite::run(a),
        Cmd::Openings {
            cmd: OpeningsCmd::Build(a),
        } => openings::run(a),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(2);
    }
}
