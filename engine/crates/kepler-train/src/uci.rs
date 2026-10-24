//! Minimal blocking UCI client. A reader thread forwards stdout lines over a channel so
//! every wait can carry a timeout, and unknown/extra lines are ignored.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    Cp(i32),
    Mate(i32),
}

impl Score {
    /// Centipawn value; mates map to +-(30000 - |plies|) style large numbers.
    pub fn to_cp(self) -> i32 {
        match self {
            Score::Cp(c) => c,
            Score::Mate(m) if m > 0 => 30_000 - m,
            Score::Mate(m) => -30_000 - m,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineSpec {
    pub cmd: Vec<String>,
    pub dir: Option<PathBuf>,
    pub opts: Vec<(String, String)>,
}

impl EngineSpec {
    /// Build from a whitespace-split command string plus extra args, a directory and
    /// `Name=Value` option strings.
    pub fn new(
        cmd: &str,
        extra: &[String],
        dir: Option<&Path>,
        opts: &[String],
    ) -> Result<Self, String> {
        let mut c: Vec<String> = cmd.split_whitespace().map(String::from).collect();
        c.extend(extra.iter().cloned());
        if c.is_empty() {
            return Err("empty engine command".into());
        }
        let mut o = Vec::new();
        for s in opts {
            let (k, v) = s
                .split_once('=')
                .ok_or_else(|| format!("bad option '{s}', expected Name=Value"))?;
            o.push((k.trim().to_string(), v.trim().to_string()));
        }
        Ok(EngineSpec {
            cmd: c,
            dir: dir.map(Path::to_path_buf),
            opts: o,
        })
    }

    pub fn plain(path: &str) -> Self {
        EngineSpec {
            cmd: vec![path.to_string()],
            dir: None,
            opts: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct GoResult {
    pub bestmove: String,
    pub score: Option<Score>,
    pub depth: Option<u32>,
}

pub struct Engine {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
    pub id_name: String,
}

impl Engine {
    #[allow(clippy::collapsible_if)]
    pub fn start(spec: &EngineSpec) -> Result<Engine, String> {
        let mut prog = PathBuf::from(&spec.cmd[0]);
        // A relative path with a separator must be resolved against *our* cwd, not --dir.
        if prog.components().count() > 1 && prog.is_relative() {
            if let Ok(abs) = std::fs::canonicalize(&prog) {
                prog = abs;
            }
        }
        let mut cmd = Command::new(&prog);
        cmd.args(&spec.cmd[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(d) = &spec.dir {
            cmd.current_dir(d);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", prog.display()))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let mut e = Engine {
            child,
            stdin,
            rx,
            id_name: String::new(),
        };
        e.send("uci")?;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let line = e
                .recv_until(deadline)
                .map_err(|m| format!("uci handshake: {m}"))?;
            if let Some(n) = line.strip_prefix("id name ") {
                e.id_name = n.trim().to_string();
            }
            if line.trim() == "uciok" {
                break;
            }
        }
        for (k, v) in &spec.opts {
            e.send(&format!("setoption name {k} value {v}"))?;
        }
        e.sync()?;
        Ok(e)
    }

    pub fn send(&mut self, s: &str) -> Result<(), String> {
        writeln!(self.stdin, "{s}")
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("write to engine failed: {e}"))
    }

    fn recv_until(&mut self, deadline: Instant) -> Result<String, String> {
        let left = deadline.saturating_duration_since(Instant::now());
        match self.rx.recv_timeout(left) {
            Ok(l) => Ok(l),
            Err(RecvTimeoutError::Timeout) => Err("timeout".into()),
            Err(RecvTimeoutError::Disconnected) => Err("engine exited".into()),
        }
    }

    /// `isready` / `readyok`, ignoring any other output.
    pub fn sync(&mut self) -> Result<(), String> {
        self.send("isready")?;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let l = self
                .recv_until(deadline)
                .map_err(|m| format!("isready: {m}"))?;
            if l.trim() == "readyok" {
                return Ok(());
            }
        }
    }

    pub fn new_game(&mut self) -> Result<(), String> {
        self.send("ucinewgame")?;
        self.sync()
    }

    /// Send `position` and `go` and wait for `bestmove`. Errors with "timeout" (after
    /// sending `stop`) if no answer within `timeout`.
    pub fn go(
        &mut self,
        position: &str,
        go_args: &str,
        timeout: Duration,
    ) -> Result<GoResult, String> {
        self.sync()?;
        self.send(position)?;
        self.send(&format!("go {go_args}"))?;
        let deadline = Instant::now() + timeout;
        let mut score = None;
        let mut depth = None;
        loop {
            let line = match self.recv_until(deadline) {
                Ok(l) => l,
                Err(m) => {
                    if m == "timeout" {
                        let _ = self.send("stop");
                    }
                    return Err(m);
                }
            };
            let mut it = line.split_whitespace();
            match it.next() {
                Some("bestmove") => {
                    let bm = it.next().unwrap_or("").to_string();
                    return Ok(GoResult {
                        bestmove: bm,
                        score,
                        depth,
                    });
                }
                Some("info") => {
                    if let Some((s, d)) = parse_info(&line) {
                        if s.is_some() {
                            score = s;
                        }
                        if d.is_some() {
                            depth = d;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Like `go`, but returns the final line of each of the first `n` principal variations
    /// (score and first move), for engines running with `MultiPV >= n`.
    pub fn go_multi(
        &mut self,
        position: &str,
        go_args: &str,
        n: usize,
        timeout: Duration,
    ) -> Result<Vec<(Score, String)>, String> {
        self.sync()?;
        self.send(position)?;
        self.send(&format!("go {go_args}"))?;
        let deadline = Instant::now() + timeout;
        let mut lines: Vec<Option<(Score, String)>> = vec![None; n];
        loop {
            let line = match self.recv_until(deadline) {
                Ok(l) => l,
                Err(m) => {
                    if m == "timeout" {
                        let _ = self.send("stop");
                    }
                    return Err(m);
                }
            };
            let t: Vec<&str> = line.split_whitespace().collect();
            match t.first() {
                Some(&"bestmove") => return Ok(lines.into_iter().flatten().collect()),
                Some(&"info") => {
                    let at = |k: &str| t.iter().position(|x| *x == k);
                    let idx = at("multipv")
                        .and_then(|i| t.get(i + 1))
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(1);
                    let score = at("score").and_then(|i| {
                        let v = t.get(i + 2)?.parse::<i32>().ok()?;
                        match *t.get(i + 1)? {
                            "cp" => Some(Score::Cp(v)),
                            "mate" => Some(Score::Mate(v)),
                            _ => None,
                        }
                    });
                    let mv = at("pv").and_then(|i| t.get(i + 1)).map(|s| s.to_string());
                    if let (Some(sc), Some(mv)) = (score, mv)
                        && idx >= 1
                        && idx <= n
                    {
                        lines[idx - 1] = Some((sc, mv));
                    }
                }
                _ => {}
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let end = Instant::now() + Duration::from_millis(300);
        while Instant::now() < end {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parse an `info` line into (score, depth). Secondary multipv lines are ignored.
pub fn parse_info(line: &str) -> Option<(Option<Score>, Option<u32>)> {
    let t: Vec<&str> = line.split_whitespace().collect();
    let mut score = None;
    let mut depth = None;
    let mut i = 1;
    while i < t.len() {
        match t[i] {
            "multipv" => {
                if t.get(i + 1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(1)
                    > 1
                {
                    return None;
                }
            }
            "depth" => depth = t.get(i + 1).and_then(|s| s.parse().ok()),
            "score" => {
                let v = t.get(i + 2).and_then(|s| s.parse::<i32>().ok());
                match (t.get(i + 1), v) {
                    (Some(&"cp"), Some(v)) => score = Some(Score::Cp(v)),
                    (Some(&"mate"), Some(v)) => score = Some(Score::Mate(v)),
                    _ => {}
                }
            }
            "pv" | "string" => break,
            _ => {}
        }
        i += 1;
    }
    Some((score, depth))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_parse() {
        let (s, d) =
            parse_info("info depth 12 seldepth 20 multipv 1 score cp -35 nodes 5 pv e2e4").unwrap();
        assert_eq!(s, Some(Score::Cp(-35)));
        assert_eq!(d, Some(12));
        let (s, _) = parse_info("info depth 3 score mate 2 pv a1a2").unwrap();
        assert_eq!(s, Some(Score::Mate(2)));
        assert!(parse_info("info depth 3 multipv 2 score cp 1 pv a1a2").is_none());
        let (s, _) = parse_info("info string score cp 5").unwrap();
        assert_eq!(s, None);
    }

    #[test]
    fn mate_cp() {
        assert!(Score::Mate(3).to_cp() > 1000);
        assert!(Score::Mate(-3).to_cp() < -1000);
    }
}
