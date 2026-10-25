//! UCI protocol front end for Kepler-64 v1.

use kepler_core::params::{Generation, SPEC_ID};
use kepler_core::search::{Limits, SearchResult, Searcher, move_from_uci, move_to_uci};
use kepler_core::{Params, Position, evaluate};
use std::io::{BufRead, Write};

struct Engine {
    pos: Position,
    history: Vec<u64>,
    params: Params,
    searcher: Searcher,
    depth_opt: u32,
    show_breakdown: bool,
}

fn score_str(r: &SearchResult) -> String {
    match r.mate_in {
        Some(m) => format!("mate {m}"),
        None => format!("cp {}", (100.0 * r.score_ku).round() as i64),
    }
}

fn info_line(board: &kepler_core::cozy_chess::Board, r: &SearchResult) -> String {
    let mut b = board.clone();
    let mut pv = Vec::new();
    for &m in &r.pv {
        // Never trust a PV blindly: stop at the first move that isn't legal.
        if !b.is_legal(m) {
            break;
        }
        pv.push(move_to_uci(&b, m));
        b.play_unchecked(m);
    }
    let nps = r.nodes * 1000 / r.elapsed_ms.max(1);
    format!(
        "info depth {} seldepth {} nodes {} nps {} time {} score {} pv {}",
        r.depth,
        r.seldepth,
        r.nodes,
        nps,
        r.elapsed_ms,
        score_str(r),
        pv.join(" ")
    )
}

impl Engine {
    fn set_position(&mut self, args: &[&str]) {
        let i;
        let pos0;
        match args.first() {
            Some(&"startpos") => {
                pos0 = Position::startpos();
                i = 1;
            }
            Some(&"fen") => {
                let end = args
                    .iter()
                    .position(|&a| a == "moves")
                    .unwrap_or(args.len());
                let fen = args[1..end].join(" ");
                match Position::from_fen(&fen) {
                    Ok(p) => pos0 = p,
                    Err(_) => return,
                }
                i = end;
            }
            _ => return,
        }
        let mut pos = pos0;
        let mut history = Vec::new();
        if args.get(i) == Some(&"moves") {
            for s in &args[i + 1..] {
                let Some(mv) = move_from_uci(&pos.board, s) else {
                    break;
                };
                history.push(pos.board.hash());
                pos = pos.play(mv, &self.params);
            }
        }
        self.pos = pos;
        self.history = history;
    }

    fn set_option(&mut self, args: &[&str], out: &mut impl Write) {
        // setoption name <name...> [value <value...>]
        let Some(ni) = args.iter().position(|&a| a == "name") else {
            return;
        };
        let vi = args.iter().position(|&a| a == "value");
        let name = args[ni + 1..vi.unwrap_or(args.len())]
            .join(" ")
            .to_lowercase();
        let value = vi.map(|v| args[v + 1..].join(" ")).unwrap_or_default();
        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.trim().parse::<usize>() {
                    self.searcher.set_hash(mb.clamp(1, 1024));
                }
            }
            "depth" => self.depth_opt = value.trim().parse::<u32>().unwrap_or(0).min(64),
            "generation" => {
                let v = value.trim();
                if v.is_empty() {
                    self.params = Params::default();
                } else {
                    match std::fs::read_to_string(v)
                        .map_err(|e| e.to_string())
                        .and_then(|s| Generation::from_json(&s).map_err(|e| e.to_string()))
                    {
                        Ok(g) => self.params = g.constants,
                        Err(e) => {
                            let _ = writeln!(out, "info string cannot load generation: {e}");
                        }
                    }
                }
            }
            "uci_showbreakdown" => self.show_breakdown = value.trim().eq_ignore_ascii_case("true"),
            _ => {}
        }
    }

    fn go(&mut self, args: &[&str], out: &mut impl Write) {
        let mut lim = Limits::default();
        let mut infinite = false;
        let mut i = 0;
        while i < args.len() {
            let num = || args.get(i + 1).and_then(|s| s.parse::<u64>().ok());
            match args[i] {
                "depth" => lim.depth = num().map(|n| n as u32),
                "nodes" => lim.nodes = num(),
                "movetime" => lim.movetime_ms = num(),
                "wtime" => lim.wtime_ms = num(),
                "btime" => lim.btime_ms = num(),
                "winc" => lim.winc_ms = num(),
                "binc" => lim.binc_ms = num(),
                "infinite" => {
                    infinite = true;
                    i += 1;
                    continue;
                }
                _ => {
                    i += 1;
                    continue;
                }
            }
            i += 2;
        }
        let _ = infinite; // no async stop in M1: an unbounded `go` runs to the depth cap
        if lim.depth.is_none() && self.depth_opt > 0 {
            lim.depth = Some(self.depth_opt);
        }
        let root = self.pos.clone();
        let board = root.board.clone();
        let mut lines = Vec::new();
        let r = self
            .searcher
            .search(&root, &self.history, &self.params, &lim, &mut |r| {
                lines.push(info_line(&board, r));
            });
        for l in lines {
            let _ = writeln!(out, "{l}");
        }
        if self.show_breakdown {
            let _ = writeln!(out, "info string breakdown {}", self.breakdown_json());
        }
        match r.best {
            Some(m) => {
                let _ = writeln!(out, "bestmove {}", move_to_uci(&board, m));
            }
            None => {
                let _ = writeln!(out, "bestmove 0000");
            }
        }
        let _ = out.flush();
    }

    fn breakdown_json(&self) -> String {
        use kepler_core::cozy_chess::Color;
        let b = evaluate::<f32>(&self.pos, &self.params);
        let v = serde_json::json!({
            "material": b.material,
            "bind": b.bind,
            "space": b.space,
            "roche": b.roche,
            "score_white": b.score_white,
            "white_king": b.king_report(Color::White, &self.params),
            "black_king": b.king_report(Color::Black, &self.params),
        });
        v.to_string()
    }
}

/// Run the UCI loop until `quit` or end of input.
pub fn run(input: impl BufRead, output: &mut impl Write) {
    let mut e = Engine {
        pos: Position::startpos(),
        history: Vec::new(),
        params: Params::default(),
        searcher: Searcher::new(64),
        depth_opt: 0,
        show_breakdown: false,
    };
    for line in input.lines() {
        let Ok(line) = line else { break };
        let toks: Vec<&str> = line.split_whitespace().collect();
        let Some((&cmd, args)) = toks.split_first() else {
            continue;
        };
        match cmd {
            "uci" => {
                let _ = writeln!(output, "id name Kepler-64 v1 ({SPEC_ID})");
                let _ = writeln!(output, "id author Ripuranjan Baruah");
                let _ = writeln!(
                    output,
                    "option name Hash type spin default 64 min 1 max 1024"
                );
                let _ = writeln!(output, "option name Depth type spin default 0 min 0 max 64");
                let _ = writeln!(output, "option name Generation type string default <empty>");
                let _ = writeln!(
                    output,
                    "option name UCI_ShowBreakdown type check default false"
                );
                let _ = writeln!(output, "uciok");
            }
            "isready" => {
                let _ = writeln!(output, "readyok");
            }
            "ucinewgame" => {
                e.searcher.clear();
                e.pos = Position::startpos();
                e.history.clear();
            }
            "setoption" => e.set_option(args, output),
            "position" => e.set_position(args),
            "go" => e.go(args, output),
            "stop" => {}
            "quit" => break,
            _ => {}
        }
        let _ = output.flush();
    }
}
