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

/// Expected score of a player `elo` points stronger: `1 / (1 + 10^(-elo/400))`.
pub fn logistic_score(elo: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-elo / 400.0))
}

/// Per-pair normalised scores of the five pentanomial outcomes (0, 0.5, 1, 1.5, 2 points / 2).
const PENTA_SCORE: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Mean and per-pair variance of the normalised pair score. Zero counts are replaced by
/// 1e-3 (fishtest's `regularize`) so a sample without variance does not divide by zero.
fn penta_moments(c: &[u32; 5]) -> (f64, f64, f64) {
    let r: Vec<f64> = c
        .iter()
        .map(|&x| if x == 0 { 1e-3 } else { x as f64 })
        .collect();
    let n: f64 = r.iter().sum();
    let mean = r.iter().zip(PENTA_SCORE).map(|(x, s)| x * s).sum::<f64>() / n;
    let var = r
        .iter()
        .zip(PENTA_SCORE)
        .map(|(x, s)| x * (s - mean).powi(2))
        .sum::<f64>()
        / n;
    (n, mean, var)
}

/// Pentanomial GSPRT log-likelihood ratio, normal approximation (the formula used by
/// fishtest's `LLR_logistic` / cutechess `-sprt`): with `N` pairs, pair-score mean `x` and
/// variance `v`, and `s_i = logistic_score(elo_i)`,
///
/// ```text
/// LLR = N * (s1 - s0) * (2 x - s0 - s1) / (2 v)
/// ```
pub fn llr_penta(c: &[u32; 5], elo0: f64, elo1: f64) -> f64 {
    if c.iter().all(|&x| x == 0) {
        return 0.0;
    }
    let (n, x, v) = penta_moments(c);
    let (s0, s1) = (logistic_score(elo0), logistic_score(elo1));
    n * (s1 - s0) * (2.0 * x - s0 - s1) / (2.0 * v)
}

/// Elo with a 95% interval from the pentanomial pair statistics (pairs are the
/// independent samples, so this is tighter and more honest than the W/D/L interval).
pub fn elo_penta(c: &[u32; 5]) -> (f64, f64, f64) {
    if c.iter().all(|&x| x == 0) {
        return (0.0, 0.0, 0.0);
    }
    let (n, x, v) = penta_moments(c);
    let se = (v / n).sqrt();
    (
        elo_from_score(x),
        elo_from_score(x - 1.96 * se),
        elo_from_score(x + 1.96 * se),
    )
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
    fn llr_by_hand() {
        // counts [10,20,40,20,10]: N=100, mean 0.5, var = (10*.25+20*.0625)*2/100 = 0.075.
        // s0 = 0.5, s1 = 1/(1+10^-0.025) = 0.514389 (10^0.025 = 1.059254).
        // LLR = 100 * 0.014389 * (1 - 0.5 - 0.514389) / (2*0.075) = -0.020704/0.15 = -0.13803
        let l = llr_penta(&[10, 20, 40, 20, 10], 0.0, 10.0);
        assert!((l + 0.13803).abs() < 2e-4, "{l}");
        // counts [5,10,30,35,20]: mean = 63.75/100 = 0.6375; E[x^2] = 0.478125;
        // var = 0.478125 - 0.6375^2 = 0.07171875; 2x - s0 - s1 = 0.260611
        // LLR = 100 * 0.014389 * 0.260611 / (2*0.07171875) = 2.6143
        let l = llr_penta(&[5, 10, 30, 35, 20], 0.0, 10.0);
        assert!((l - 2.6143).abs() < 5e-3, "{l}");
        // bounds for alpha = beta = 0.05 are +-ln(19) = +-2.9444
        assert!((19f64.ln() - 2.9444).abs() < 1e-3);
        assert_eq!(llr_penta(&[0; 5], 0.0, 10.0), 0.0);
        // no variance at all stays finite
        assert!(llr_penta(&[0, 0, 0, 0, 50], 0.0, 10.0).is_finite());
        let (e, lo, hi) = elo_penta(&[10, 20, 40, 20, 10]);
        assert!(e.abs() < 1e-9 && lo < 0.0 && hi > 0.0);
    }

    #[test]
    fn penta() {
        assert_eq!(
            pentanomial(&[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0]),
            [1, 1, 2, 1, 3]
        );
    }
}
