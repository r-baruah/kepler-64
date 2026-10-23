//! kepler-train: testing and training harness for Kepler-64 v1.

mod chess;
mod epd;
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
    /// SPRT between two generation files (in-process self-play)
    Sprt(sprt::SprtArgs),
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
        Cmd::Sprt(a) => sprt::run(a),
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
