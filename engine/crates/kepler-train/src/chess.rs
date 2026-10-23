//! Chess helpers on top of cozy-chess: UCI move conversion (castling!), SAN parse and
//! write, and game-end detection.
//!
//! cozy-chess encodes castling as "king takes own rook" (e1h1). UCI engines use e1g1.

use cozy_chess::{Board, Color, File, GameStatus, Move, Piece, Rank, Square};

pub fn legal_moves(b: &Board) -> Vec<Move> {
    let mut v = Vec::new();
    b.generate_moves(|pm| {
        v.extend(pm);
        false
    });
    v
}

/// Normalise an opening/EPD position to `board stm castling ep 0 1` for engines.
pub fn fen_with_counters(fen4: &str) -> String {
    format!("{fen4} 0 1")
}

pub fn board_from_fen4(fen4: &str) -> Result<Board, String> {
    Board::from_fen(&fen_with_counters(fen4), false).map_err(|e| format!("bad FEN '{fen4}': {e:?}"))
}

fn is_castle(b: &Board, mv: Move) -> bool {
    b.piece_on(mv.from) == Some(Piece::King) && b.color_on(mv.to) == Some(b.side_to_move())
}

/// cozy move -> UCI string (castling as e1g1).
#[cfg_attr(not(test), allow(dead_code))]
pub fn move_to_uci(b: &Board, mv: Move) -> String {
    if is_castle(b, mv) {
        let file = if mv.to.file() > mv.from.file() {
            File::G
        } else {
            File::C
        };
        return format!("{}{}", mv.from, Square::new(file, mv.from.rank()));
    }
    mv.to_string()
}

/// UCI string -> legal cozy move (converting e1g1 to king-takes-rook). Err if the text
/// is malformed or the move is illegal.
pub fn uci_to_move(b: &Board, s: &str) -> Result<Move, String> {
    let mut mv: Move = s.parse().map_err(|_| format!("unparsable move '{s}'"))?;
    let stm = b.side_to_move();
    if b.piece_on(mv.from) == Some(Piece::King)
        && mv.from.rank() == mv.to.rank()
        && (mv.from.file() as i32 - mv.to.file() as i32).abs() == 2
    {
        let rights = b.castle_rights(stm);
        let rook_file = if mv.to.file() > mv.from.file() {
            rights.short
        } else {
            rights.long
        };
        if let Some(f) = rook_file {
            mv.to = Square::new(f, mv.from.rank());
        }
    } else if b.color_on(mv.to) == Some(stm) {
        return Err(format!("'{s}' captures own piece"));
    }
    if b.is_legal(mv) {
        Ok(mv)
    } else {
        Err(format!("illegal move '{s}'"))
    }
}

fn piece_letter(p: Piece) -> Option<char> {
    match p {
        Piece::Pawn => None,
        Piece::Knight => Some('N'),
        Piece::Bishop => Some('B'),
        Piece::Rook => Some('R'),
        Piece::Queen => Some('Q'),
        Piece::King => Some('K'),
    }
}

fn file_char(f: File) -> char {
    (b'a' + f as u8) as char
}

fn rank_char(r: Rank) -> char {
    (b'1' + r as u8) as char
}

/// Write a legal move in SAN (with +/# suffix).
pub fn move_to_san(b: &Board, mv: Move) -> String {
    let piece = b.piece_on(mv.from).unwrap_or(Piece::Pawn);
    let mut s = String::new();
    if is_castle(b, mv) {
        s.push_str(if mv.to.file() > mv.from.file() {
            "O-O"
        } else {
            "O-O-O"
        });
    } else {
        let capture = b.color_on(mv.to) == Some(!b.side_to_move())
            || (piece == Piece::Pawn && mv.from.file() != mv.to.file());
        if let Some(c) = piece_letter(piece) {
            s.push(c);
            let others: Vec<Move> = legal_moves(b)
                .into_iter()
                .filter(|m| m.to == mv.to && m.from != mv.from && b.piece_on(m.from) == Some(piece))
                .collect();
            if !others.is_empty() {
                let same_file = others.iter().any(|m| m.from.file() == mv.from.file());
                let same_rank = others.iter().any(|m| m.from.rank() == mv.from.rank());
                if !same_file {
                    s.push(file_char(mv.from.file()));
                } else if !same_rank {
                    s.push(rank_char(mv.from.rank()));
                } else {
                    s.push(file_char(mv.from.file()));
                    s.push(rank_char(mv.from.rank()));
                }
            }
        } else if capture {
            s.push(file_char(mv.from.file()));
        }
        if capture {
            s.push('x');
        }
        s.push(file_char(mv.to.file()));
        s.push(rank_char(mv.to.rank()));
        if let Some(p) = mv.promotion {
            s.push('=');
            s.push(piece_letter(p).unwrap_or('Q'));
        }
    }
    let mut next = b.clone();
    next.play(mv);
    if !next.checkers().is_empty() {
        s.push(if legal_moves(&next).is_empty() {
            '#'
        } else {
            '+'
        });
    }
    s
}

/// Parse SAN (piece letter, disambiguation, captures, `=Q`, O-O / O-O-O, check marks).
#[allow(clippy::collapsible_if)]
pub fn san_to_move(b: &Board, san: &str) -> Result<Move, String> {
    let t: String = san
        .trim()
        .trim_end_matches(['+', '#', '!', '?'])
        .replace("e.p.", "")
        .replace('0', "O");
    let stm = b.side_to_move();
    let moves = legal_moves(b);
    if t == "O-O" || t == "O-O-O" {
        let short = t == "O-O";
        return moves
            .into_iter()
            .find(|m| is_castle(b, *m) && (m.to.file() > m.from.file()) == short)
            .ok_or_else(|| format!("castling '{san}' not legal"));
    }
    let chars: Vec<char> = t.chars().collect();
    if chars.len() < 2 {
        return Err(format!("bad SAN '{san}'"));
    }
    let mut i = 0;
    let piece = match chars[0] {
        'N' => Piece::Knight,
        'B' => Piece::Bishop,
        'R' => Piece::Rook,
        'Q' => Piece::Queen,
        'K' => Piece::King,
        _ => Piece::Pawn,
    };
    if piece != Piece::Pawn {
        i = 1;
    }
    // Promotion suffix: "=Q" or trailing letter.
    let mut end = chars.len();
    let mut promo = None;
    if end >= 2 {
        let c = chars[end - 1];
        let p = match c {
            'N' => Some(Piece::Knight),
            'B' => Some(Piece::Bishop),
            'R' => Some(Piece::Rook),
            'Q' => Some(Piece::Queen),
            _ => None,
        };
        if let Some(p) = p {
            if piece == Piece::Pawn {
                promo = Some(p);
                end -= 1;
                if end > 0 && chars[end - 1] == '=' {
                    end -= 1;
                }
            }
        }
    }
    let body: Vec<char> = chars[i..end]
        .iter()
        .copied()
        .filter(|c| *c != 'x' && *c != '-')
        .collect();
    if body.len() < 2 {
        return Err(format!("bad SAN '{san}'"));
    }
    let dest_f = body[body.len() - 2];
    let dest_r = body[body.len() - 1];
    if !('a'..='h').contains(&dest_f) || !('1'..='8').contains(&dest_r) {
        return Err(format!("bad destination in SAN '{san}'"));
    }
    let to: Square = format!("{dest_f}{dest_r}")
        .parse()
        .map_err(|_| format!("bad square in '{san}'"))?;
    let mut from_file = None;
    let mut from_rank = None;
    for c in &body[..body.len() - 2] {
        if ('a'..='h').contains(c) {
            from_file = Some(*c);
        } else if ('1'..='8').contains(c) {
            from_rank = Some(*c);
        } else {
            return Err(format!("bad disambiguation in '{san}'"));
        }
    }
    let cands: Vec<Move> = moves
        .into_iter()
        .filter(|m| {
            b.piece_on(m.from) == Some(piece)
                && m.to == to
                && !is_castle(b, *m)
                && m.promotion == promo
                && from_file.is_none_or(|f| file_char(m.from.file()) == f)
                && from_rank.is_none_or(|r| rank_char(m.from.rank()) == r)
                // pawn pushes without a file letter must not be captures
                && !(piece == Piece::Pawn && from_file.is_none() && m.from.file() != to.file())
        })
        .collect();
    let _ = stm;
    match cands.len() {
        1 => Ok(cands[0]),
        0 => Err(format!("no legal move matches SAN '{san}'")),
        _ => Err(format!("ambiguous SAN '{san}'")),
    }
}

fn insufficient_material(b: &Board) -> bool {
    if !(b.pieces(Piece::Pawn) | b.pieces(Piece::Rook) | b.pieces(Piece::Queen)).is_empty() {
        return false;
    }
    let knights = b.pieces(Piece::Knight).len() as usize;
    let bishops = b.pieces(Piece::Bishop);
    if knights + bishops.len() as usize <= 1 {
        return true;
    }
    if knights == 0 {
        let mut light = false;
        let mut dark = false;
        for sq in bishops {
            if (sq.file() as u8 + sq.rank() as u8) % 2 == 1 {
                light = true;
            } else {
                dark = true;
            }
        }
        return !(light && dark);
    }
    false
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum End {
    WhiteWins,
    BlackWins,
    Draw,
}

/// Game-over check. `history` holds every position since the start including `b` last.
pub fn game_end(b: &Board, history: &[Board]) -> Option<(End, &'static str)> {
    match b.status() {
        GameStatus::Won => {
            let e = if b.side_to_move() == Color::White {
                End::BlackWins
            } else {
                End::WhiteWins
            };
            return Some((e, "checkmate"));
        }
        GameStatus::Drawn => {
            let r = if legal_moves(b).is_empty() {
                "stalemate"
            } else {
                "fifty_move_rule"
            };
            return Some((End::Draw, r));
        }
        GameStatus::Ongoing => {}
    }
    if insufficient_material(b) {
        return Some((End::Draw, "insufficient_material"));
    }
    if history.iter().filter(|h| h.same_position(b)).count() >= 3 {
        return Some((End::Draw, "threefold_repetition"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(fen: &str) -> Board {
        Board::from_fen(fen, false).unwrap()
    }

    #[test]
    fn castling_uci_roundtrip() {
        let bd = b("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let mv = uci_to_move(&bd, "e1g1").unwrap();
        assert_eq!(mv.to_string(), "e1h1");
        assert_eq!(move_to_uci(&bd, mv), "e1g1");
        let mv = uci_to_move(&bd, "e1c1").unwrap();
        assert_eq!(move_to_uci(&bd, mv), "e1c1");
        // king-takes-rook text is not valid UCI
        assert!(uci_to_move(&bd, "e1h1").is_err());
        let bd = b("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
        let mv = uci_to_move(&bd, "e8g8").unwrap();
        assert_eq!(move_to_uci(&bd, mv), "e8g8");
        // plain king move unaffected, illegal move rejected
        assert!(uci_to_move(&bd, "e8e7").is_ok());
        assert!(uci_to_move(&bd, "e8e6").is_err());
        // no castling rights -> e1g1 illegal
        let bd = b("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
        assert!(uci_to_move(&bd, "e1g1").is_err());
    }

    #[test]
    fn san_basic_and_castling() {
        let bd = Board::default();
        let mv = san_to_move(&bd, "Nf3").unwrap();
        assert_eq!(mv.to_string(), "g1f3");
        assert_eq!(san_to_move(&bd, "e4").unwrap().to_string(), "e2e4");
        assert!(san_to_move(&bd, "e5").is_err());
        let bd = b("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(move_to_uci(&bd, san_to_move(&bd, "O-O").unwrap()), "e1g1");
        assert_eq!(
            move_to_uci(&bd, san_to_move(&bd, "O-O-O+").unwrap()),
            "e1c1"
        );
        assert_eq!(move_to_uci(&bd, san_to_move(&bd, "0-0").unwrap()), "e1g1");
        let bd = b("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
        assert_eq!(move_to_uci(&bd, san_to_move(&bd, "O-O-O").unwrap()), "e8c8");
        assert_eq!(move_to_san(&bd, san_to_move(&bd, "O-O").unwrap()), "O-O");
    }

    #[test]
    fn san_promotion_and_capture() {
        let bd = b("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        assert_eq!(san_to_move(&bd, "a8=Q").unwrap().to_string(), "a7a8q");
        assert_eq!(san_to_move(&bd, "a8=N+").unwrap().to_string(), "a7a8n");
        assert_eq!(san_to_move(&bd, "axb8=R").unwrap().to_string(), "a7b8r");
        assert_eq!(san_to_move(&bd, "axb8=Q").unwrap().to_string(), "a7b8q");
        assert!(san_to_move(&bd, "a8").is_err());
        let mv = san_to_move(&bd, "a8=Q").unwrap();
        assert_eq!(move_to_san(&bd, mv), "a8=Q");
    }

    #[test]
    fn san_disambiguation() {
        // two knights can reach d2; rooks on a1/a3 can reach a2; rooks a1/h1 reach d1.
        let bd = b("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1");
        assert_eq!(san_to_move(&bd, "Nbd2").unwrap().to_string(), "b1d2");
        assert_eq!(san_to_move(&bd, "Nfd2").unwrap().to_string(), "f3d2");
        assert!(san_to_move(&bd, "Nd2").is_err());
        assert_eq!(move_to_san(&bd, "b1d2".parse().unwrap()), "Nbd2");
        let bd = b("4k3/8/8/8/R7/8/8/R3K3 w - - 0 1");
        assert_eq!(san_to_move(&bd, "R1a2").unwrap().to_string(), "a1a2");
        assert_eq!(san_to_move(&bd, "R4a2").unwrap().to_string(), "a4a2");
        assert_eq!(move_to_san(&bd, "a1a2".parse().unwrap()), "R1a2");
        // three queens: full square disambiguation
        let bd = b("6k1/8/8/8/Q6Q/8/8/Q3K3 w - - 0 1");
        assert_eq!(move_to_san(&bd, "a4d4".parse().unwrap()), "Qa4d4");
    }

    #[test]
    fn san_pawn_captures_and_ep() {
        let bd = b("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        assert_eq!(san_to_move(&bd, "exd6").unwrap().to_string(), "e5d6");
        assert_eq!(move_to_san(&bd, "e5d6".parse().unwrap()), "exd6");
        assert_eq!(san_to_move(&bd, "e6").unwrap().to_string(), "e5e6");
        let bd = b("4k3/8/8/8/8/8/8/R3K2R w - - 0 1");
        assert_eq!(move_to_san(&bd, "a1a8".parse().unwrap()), "Ra8+");
    }

    #[test]
    fn game_ends() {
        let bd = b("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        assert_eq!(
            game_end(&bd, std::slice::from_ref(&bd)).unwrap().1,
            "stalemate"
        );
        let bd = b("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1");
        assert_eq!(
            game_end(&bd, std::slice::from_ref(&bd)).unwrap().0,
            End::WhiteWins
        );
        let bd = b("4k3/8/8/8/8/8/8/4KB2 w - - 0 1");
        assert_eq!(
            game_end(&bd, std::slice::from_ref(&bd)).unwrap().1,
            "insufficient_material"
        );
        let bd = b("4k3/8/8/8/8/8/8/3RK3 w - - 0 1");
        assert!(game_end(&bd, std::slice::from_ref(&bd)).is_none());
        // threefold
        let mut cur = Board::default();
        let mut hist = vec![cur.clone()];
        for m in [
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ] {
            cur.play(m.parse().unwrap());
            hist.push(cur.clone());
        }
        assert_eq!(game_end(&cur, &hist).unwrap().1, "threefold_repetition");
        // fifty-move
        let bd = b("4k3/8/8/8/8/8/8/3RK3 w - - 100 80");
        assert_eq!(
            game_end(&bd, std::slice::from_ref(&bd)).unwrap().1,
            "fifty_move_rule"
        );
    }
}
