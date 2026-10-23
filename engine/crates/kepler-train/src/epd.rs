//! EPD / FEN line parsing.

pub struct Epd {
    /// First four FEN fields (board, side, castling, ep).
    pub fen4: String,
    pub ops: Vec<(String, String)>,
}

impl Epd {
    pub fn op(&self, name: &str) -> Option<&str> {
        self.ops
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

fn take_token(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

/// Parse one line; `None` for blank lines and `#` comments.
pub fn parse_line(line: &str) -> Option<Epd> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let mut rest = line;
    let mut f = Vec::new();
    for _ in 0..4 {
        let (t, r) = take_token(rest);
        if t.is_empty() {
            return None;
        }
        f.push(t);
        rest = r;
    }
    // optional "halfmove fullmove" counters of a full FEN
    for _ in 0..2 {
        let (t, r) = take_token(rest);
        if !t.is_empty() && t.trim_end_matches(';').chars().all(|c| c.is_ascii_digit()) {
            rest = if t.ends_with(';') {
                &line[line.len() - r.len() - 1..]
            } else {
                r
            };
        } else {
            break;
        }
    }
    let mut ops = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut pieces = Vec::new();
    for c in rest.chars() {
        match c {
            '"' => {
                in_q = !in_q;
                cur.push(c);
            }
            ';' if !in_q => pieces.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    pieces.push(cur);
    for p in pieces {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        let (k, v) = take_token(p);
        ops.push((k.to_string(), v.trim().trim_matches('"').to_string()));
    }
    Some(Epd {
        fen4: f.join(" "),
        ops,
    })
}

pub fn parse_file(path: &std::path::Path) -> Result<Vec<Epd>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text.lines().filter_map(parse_line).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epd_ops() {
        let e = parse_line(r#"r1b1kbnr/pp3ppp/1qn1p3/3pP3/2pP4/2PQ1N2/PP3PPP/RNB1KB1R w KQkq - 0 7 am Qxc4 Qd1; bm Qc2; id "french w"; c0 "a; b";"#).unwrap();
        assert_eq!(
            e.fen4,
            "r1b1kbnr/pp3ppp/1qn1p3/3pP3/2pP4/2PQ1N2/PP3PPP/RNB1KB1R w KQkq -"
        );
        assert_eq!(e.op("am"), Some("Qxc4 Qd1"));
        assert_eq!(e.op("bm"), Some("Qc2"));
        assert_eq!(e.op("id"), Some("french w"));
        assert_eq!(e.op("c0"), Some("a; b"));
        let e = parse_line("8/8/8/8/8/8/8/K6k w - - bm Ka2;").unwrap();
        assert_eq!(e.op("bm"), Some("Ka2"));
        let e = parse_line("8/8/8/8/8/8/8/K6k w - - 0 1 ; id \"x\"").unwrap();
        assert_eq!(e.op("id"), Some("x"));
        assert!(parse_line("# hi").is_none());
    }
}
