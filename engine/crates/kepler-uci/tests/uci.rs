use kepler_core::Position;
use kepler_core::search::move_from_uci;

fn drive(script: &str) -> String {
    let mut out = Vec::new();
    kepler_uci::run(script.as_bytes(), &mut out);
    String::from_utf8(out).unwrap()
}

fn bestmove(out: &str) -> String {
    out.lines()
        .rev()
        .find_map(|l| l.strip_prefix("bestmove "))
        .expect("bestmove line")
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

#[test]
fn smoke() {
    let out = drive("uci\nisready\nposition startpos moves e2e4\ngo depth 4\nquit\n");
    assert!(out.contains("uciok"));
    assert!(out.contains("readyok"));
    assert!(out.contains("id name Kepler-64 v1 (v1.0-spec-"));
    assert!(out.contains("info depth 4"));
    let mut pos = Position::startpos();
    pos.board.play(move_from_uci(&pos.board, "e2e4").unwrap());
    let bm = bestmove(&out);
    assert!(
        move_from_uci(&pos.board, &bm).is_some(),
        "illegal bestmove {bm}"
    );
}

#[test]
fn castling_roundtrip() {
    let fen = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1";
    let out = drive(&format!("position fen {fen}\ngo depth 3\nquit\n"));
    let pos = Position::from_fen(fen).unwrap();
    let bm = bestmove(&out);
    assert!(
        move_from_uci(&pos.board, &bm).is_some(),
        "illegal bestmove {bm}"
    );

    // `moves e1g1` must be parsed as castling and the rook must land on f1.
    let script = format!("position fen {fen} moves e1g1\ngo depth 2\nquit\n");
    let out = drive(&script);
    let mut after = pos.clone();
    after.board.play(move_from_uci(&pos.board, "e1g1").unwrap());
    assert_eq!(
        after
            .board
            .king(kepler_core::cozy_chess::Color::White)
            .to_string(),
        "g1"
    );
    let bm = bestmove(&out);
    assert!(
        move_from_uci(&after.board, &bm).is_some(),
        "illegal reply {bm}"
    );

    // Castling is printed in UCI form, never as king-takes-rook.
    let castle = "position fen 4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1\ngo depth 1\nquit\n";
    let out = drive(castle);
    let bm = bestmove(&out);
    assert!(!matches!(bm.as_str(), "e1h1" | "e1a1"));
}

#[test]
fn breakdown_option() {
    let out =
        drive("setoption name UCI_ShowBreakdown value true\nposition startpos\ngo depth 1\nquit\n");
    assert!(out.contains("info string breakdown {"));
    assert!(out.contains("white_king"));
}
