//! Elo and pentanomial statistics.

use serde::Serialize;

/// Logistic Elo from a score fraction: `elo = -400 * log10(1/s - 1)`. `s` is clamped to
/// [0.001, 0.999] so all-win / all-loss samples give finite numbers.
pub fn elo_from_score(s: f64) -> f64 {
    let s = s.clamp(0.001, 0.999);
    -400.0 * (1.0 / s - 1.0).log10()
}

#[derive(Serialize, Debug, Clone, Copy)]
pub struct Elo {
    pub estimate: f64,
    pub lo95: f64,
    pub hi95: f64,
    pub score: f64,
    pub games: u32,
    /// false when fewer than 200 games (docs/v1/04: never report an Elo from < 200 games)
    pub reliable: bool,
}

/// Elo estimate with a 95% interval from the trinomial (W/D/L) variance:
///
/// ```text
/// n    = w + d + l
/// s    = (w + d/2) / n
/// var  = (w*(1-s)^2 + d*(0.5-s)^2 + l*s^2) / n        (per-game score variance)
/// se   = sqrt(var / n)
/// elo  = -400 log10(1/s - 1);   lo/hi = elo(s -/+ 1.96 se)
/// ```
pub fn elo(w: u32, d: u32, l: u32) -> Elo {
    let n = (w + d + l) as f64;
    if n == 0.0 {
        return Elo {
            estimate: 0.0,
            lo95: 0.0,
            hi95: 0.0,
            score: 0.5,
            games: 0,
            reliable: false,
        };
    }
    let (wf, df, lf) = (w as f64, d as f64, l as f64);
    let s = (wf + df / 2.0) / n;
    let var = (wf * (1.0 - s).powi(2) + df * (0.5 - s).powi(2) + lf * s.powi(2)) / n;
    let se = (var / n).sqrt();
    Elo {
        estimate: elo_from_score(s),
        lo95: elo_from_score(s - 1.96 * se),
        hi95: elo_from_score(s + 1.96 * se),
        score: s,
        games: n as u32,
        reliable: n >= 200.0,
    }
}

/// Pentanomial counts `[0, 0.5, 1, 1.5, 2]` points over game pairs.
pub fn pentanomial(pair_points: &[f64]) -> [u32; 5] {
    let mut p = [0u32; 5];
    for x in pair_points {
        let i = (x * 2.0).round() as usize;
        p[i.min(4)] += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_elo() {
        assert!(elo_from_score(0.5).abs() < 1e-9);
        assert!((elo_from_score(0.75) - 190.848).abs() < 0.01);
        assert!((elo_from_score(0.25) + 190.848).abs() < 0.01);
        let e = elo(50, 0, 50);
        assert!(e.estimate.abs() < 1e-9);
        assert!(e.lo95 < 0.0 && e.hi95 > 0.0);
        // se = sqrt(0.25/100) = 0.05 -> interval symmetric about 0, ~ +-69.6
        assert!((e.hi95 - 69.6).abs() < 1.0, "{}", e.hi95);
        let e = elo(75, 0, 25);
        assert!((e.estimate - 190.848).abs() < 0.01);
        assert!(!e.reliable);
        assert!(elo(150, 0, 100).reliable);
        // all draws -> zero variance, zero Elo
        let e = elo(0, 10, 0);
        assert!(e.estimate.abs() < 1e-9 && e.lo95.abs() < 1e-9);
        assert!(elo(10, 0, 0).estimate.is_finite());
    }

    #[test]
    fn penta() {
        assert_eq!(
            pentanomial(&[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]),
            [1, 1, 2, 1, 3]
        );
    }
}
