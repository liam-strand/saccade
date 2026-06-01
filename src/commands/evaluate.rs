//! Implementation of the `evaluate` subcommand: compare an estimated Perfetto trace against a ground-truth trace using nRMSE and coverage metrics.

use crate::perfetto;
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

/// Per-event evaluation results collected for a single (event, thread) pair.
struct EventMetrics {
    /// Name of the hardware event as reported by `perf list`.
    event_name: String,
    /// Thread ID the metrics apply to.
    tid: u32,
    /// Normalised RMSE between estimated and ground-truth rate series; `None` when the ground-truth
    /// mean rate is zero, or when the estimator has no bins for this event (never observed).
    /// Only GT bins at or after the estimator's first observed bin are scored — pre-first-observation
    /// GT bins are excluded from both the sum and the denominator to avoid penalising round-robin
    /// schedulers for the latency-to-first-sample, which is a scheduling-policy artifact rather
    /// than a measure of predictive skill.
    nrmse: Option<f64>,
    /// Fraction of ground-truth time bins for which an estimated value exists (0.0–1.0).
    /// Coverage is always computed over ALL GT bins (not clipped), so it captures the full
    /// scheduling lag on an independent axis from nRMSE.
    coverage: f64,
    /// Mean ground-truth event rate in events per nanosecond, used to normalise the RMSE.
    mean_gt_rate_events_per_ns: f64,
    /// GT-anchored calibration score: fraction of scored nRMSE bins where the GT rate falls
    /// within the estimator's predicted band `[est_rate * (1 - uncertainty), est_rate * (1 + uncertainty)]`.
    /// `None` when there are no scored bins (same condition as `nrmse == None`) or when
    /// no uncertainty series is available for this event.
    calibration: Option<f64>,
    /// Coefficient of variation of the GT rate series (stddev / mean) over all GT bins.
    /// Measures how dynamic/hard the event is; useful for importance-weighting in downstream analysis.
    /// `None` when mean_gt == 0.
    gt_cv: Option<f64>,
}

/// Computes per-event nRMSE and coverage by binning both traces at `bin_ms` width, then prints results as text or JSON.
pub fn evaluate(
    ground_truth: PathBuf,
    estimated: PathBuf,
    bin_ms: u64,
    json: bool,
) -> io::Result<()> {
    let mut gt = perfetto::read_rate_timeseries(&ground_truth)?;
    let mut est = perfetto::read_rate_timeseries(&estimated)?;

    // Re-anchor each trace to t=0 independently so their bin indices align.
    // GT traces from real hardware carry absolute monotonic-clock timestamps
    // (~10^18 ns); simulated traces start near 0. Without this, every GT bin
    // misses in est_bins and RMSE is computed against imputed zeros throughout.
    normalize_timestamps(&mut gt.series);
    normalize_timestamps(&mut est.series);
    // Uncertainty shares timestamps with rate (both emitted per snapshot); apply
    // the same normalization so bin indices align with the rate bins.
    normalize_timestamps(&mut est.uncertainty);

    let bin_width_ns = bin_ms * 1_000_000;

    // Warn about spurious tracks (in estimated but not in ground truth).
    let mut spurious: Vec<(String, u32)> = est
        .series
        .keys()
        .filter(|k| !gt.series.contains_key(*k))
        .cloned()
        .collect();
    if !spurious.is_empty() {
        spurious.sort();
        eprintln!(
            "Warning: {} track(s) in estimated trace not found in ground truth:",
            spurious.len()
        );
        for (name, tid) in &spurious {
            eprintln!("  {name} tid={tid}");
        }
    }

    let mut results: Vec<EventMetrics> = Vec::new();

    let mut keys: Vec<(String, u32)> = gt.series.keys().cloned().collect();
    keys.sort();

    for (event_name, tid) in &keys {
        let gt_points = &gt.series[&(event_name.clone(), *tid)];
        let gt_bins = bin_avg(gt_points, bin_width_ns);
        let est_bins = est
            .series
            .get(&(event_name.clone(), *tid))
            .map(|pts| bin_avg(pts, bin_width_ns))
            .unwrap_or_default();
        let unc_bins = est
            .uncertainty
            .get(&(event_name.clone(), *tid))
            .map(|pts| bin_avg(pts, bin_width_ns))
            .unwrap_or_default();

        let n_gt = gt_bins.len();
        if n_gt == 0 {
            continue;
        }

        // Determine the first bin index at which the estimator has made an observation.
        // GT bins before this point are excluded from nRMSE to avoid penalising scheduling
        // lag (a round-robin artifact) rather than predictive accuracy.
        let first_est_bin: Option<u64> = est_bins.keys().copied().min();

        let mean_gt = gt_bins.values().sum::<f64>() / n_gt as f64;

        // Coverage: fraction of ALL GT bins that have a corresponding estimated bin
        // (unchanged — measures scheduling lag on an independent axis).
        let covered = gt_bins.keys().filter(|b| est_bins.contains_key(b)).count();
        let coverage = covered as f64 / n_gt as f64;

        // nRMSE: scored only over GT bins at/after first_est_bin.
        // If est_bins is empty (never observed) → nrmse = None.
        // If mean_gt == 0 → nrmse = None (event never fires in GT).
        let nrmse = if mean_gt > 0.0 {
            if let Some(first_bin) = first_est_bin {
                let mut sq_sum = 0.0f64;
                let mut n_scored = 0usize;
                for (&bin, &gt_rate) in &gt_bins {
                    if bin < first_bin {
                        continue;
                    }
                    let est_rate = est_bins.get(&bin).copied().unwrap_or(0.0);
                    let rel_err = (est_rate - gt_rate) / mean_gt;
                    sq_sum += rel_err * rel_err;
                    n_scored += 1;
                }
                if n_scored > 0 {
                    Some((sq_sum / n_scored as f64).sqrt())
                } else {
                    None
                }
            } else {
                // No estimated bins at all — estimator never observed this event.
                None
            }
        } else {
            None
        };

        // GT coefficient of variation (stddev / mean) — measures signal dynamics.
        let gt_cv = if mean_gt > 0.0 {
            let variance = gt_bins
                .values()
                .map(|&v| (v - mean_gt).powi(2))
                .sum::<f64>()
                / n_gt as f64;
            Some(variance.sqrt() / mean_gt)
        } else {
            None
        };

        // GT-anchored calibration: fraction of scored bins where GT rate falls inside
        // the estimator's predicted band.  Band definition:
        //   lower = est_rate * (1 - uncertainty)
        //   upper = est_rate * (1 + uncertainty)
        // where `uncertainty` ∈ [0, 1].  When est_rate == 0 the band is [0, 0] and
        // calibration credit is given only when gt_rate is also 0 (exact match).
        // Scored over the same at/after-first-observation window as nRMSE.
        // `None` when there are no scored bins or no uncertainty data for this event.
        let calibration = match (unc_bins.is_empty(), first_est_bin) {
            (false, Some(first_bin)) => {
                let mut in_band = 0usize;
                let mut n_scored = 0usize;
                for (&bin, &gt_rate) in &gt_bins {
                    if bin < first_bin {
                        continue;
                    }
                    let est_rate = est_bins.get(&bin).copied().unwrap_or(0.0);
                    let unc = unc_bins.get(&bin).copied().unwrap_or(1.0);
                    let lower = est_rate * (1.0 - unc);
                    let upper = est_rate * (1.0 + unc);
                    if gt_rate >= lower && gt_rate <= upper {
                        in_band += 1;
                    }
                    n_scored += 1;
                }
                if n_scored > 0 {
                    Some(in_band as f64 / n_scored as f64)
                } else {
                    None
                }
            }
            _ => None,
        };

        results.push(EventMetrics {
            event_name: event_name.clone(),
            tid: *tid,
            nrmse,
            coverage,
            mean_gt_rate_events_per_ns: mean_gt,
            calibration,
            gt_cv,
        });
    }

    let nrmse_vals: Vec<f64> = results.iter().filter_map(|r| r.nrmse).collect();
    let mean_nrmse = if nrmse_vals.is_empty() {
        None
    } else {
        Some(nrmse_vals.iter().sum::<f64>() / nrmse_vals.len() as f64)
    };
    let zero_coverage_count = results.iter().filter(|r| r.coverage == 0.0).count();
    let mean_coverage = if results.is_empty() {
        None
    } else {
        Some(results.iter().map(|r| r.coverage).sum::<f64>() / results.len() as f64)
    };
    let cal_vals: Vec<f64> = results.iter().filter_map(|r| r.calibration).collect();
    let mean_calibration = if cal_vals.is_empty() {
        None
    } else {
        Some(cal_vals.iter().sum::<f64>() / cal_vals.len() as f64)
    };

    if json {
        print_json(
            &ground_truth,
            &estimated,
            bin_ms,
            &results,
            mean_nrmse,
            zero_coverage_count,
            mean_coverage,
            mean_calibration,
        );
    } else {
        print_text(
            &ground_truth,
            &estimated,
            bin_ms,
            &results,
            mean_nrmse,
            zero_coverage_count,
            mean_coverage,
            mean_calibration,
        );
    }

    Ok(())
}

/// Shifts every timestamp in `series` so the earliest point across all tracks lands at t=0.
fn normalize_timestamps<K: Eq + std::hash::Hash>(series: &mut HashMap<K, Vec<(u64, f64)>>) {
    let min_ts: u64 = series
        .values()
        .filter_map(|pts| pts.first().map(|&(ts, _)| ts))
        .min()
        .unwrap_or(0);
    if min_ts > 0 {
        for pts in series.values_mut() {
            for (ts, _) in pts.iter_mut() {
                *ts -= min_ts;
            }
        }
    }
}

/// Averages `points` into fixed-width time bins of `bin_width_ns` nanoseconds, returning a map from bin index to mean value.
fn bin_avg(points: &[(u64, f64)], bin_width_ns: u64) -> HashMap<u64, f64> {
    let mut acc: HashMap<u64, (f64, usize)> = HashMap::new();
    for &(ts, rate) in points {
        let bin = ts / bin_width_ns;
        let e = acc.entry(bin).or_default();
        e.0 += rate;
        e.1 += 1;
    }
    acc.into_iter()
        .map(|(b, (s, n))| (b, s / n as f64))
        .collect()
}

/// Serialises `v` to a JSON number string, substituting `"null"` for non-finite values that are invalid in JSON.
fn f64_to_json(v: f64) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "null".to_string()
    }
}

/// Prints evaluation results as a human-readable table to stdout.
#[allow(clippy::too_many_arguments)]
fn print_text(
    ground_truth: &std::path::Path,
    estimated: &std::path::Path,
    bin_ms: u64,
    results: &[EventMetrics],
    mean_nrmse: Option<f64>,
    zero_coverage_count: usize,
    mean_coverage: Option<f64>,
    mean_calibration: Option<f64>,
) {
    println!("Evaluation Report");
    println!("=================");
    println!("Ground truth: {}", ground_truth.display());
    println!("Estimated:    {}", estimated.display());
    println!("Bin width:    {bin_ms} ms");
    println!();
    println!(
        "{:<32} {:>5}  {:<12}  {:<16}  Coverage  Calibration  GT-CV",
        "Event", "TID", "nRMSE", "Mean GT (ev/ns)"
    );
    println!("{}", "-".repeat(100));
    for r in results {
        let nrmse_str = match r.nrmse {
            Some(v) => format!("{v:.3e}"),
            None => "—".to_string(),
        };
        let cal_str = match r.calibration {
            Some(v) => format!("{:.1}%", v * 100.0),
            None => "—".to_string(),
        };
        let cv_str = match r.gt_cv {
            Some(v) => format!("{v:.3e}"),
            None => "—".to_string(),
        };
        println!(
            "{:<32} {:>5}  {:<12}  {:<16.3e}  {:.1}%      {:<13}  {:<10}",
            r.event_name,
            r.tid,
            nrmse_str,
            r.mean_gt_rate_events_per_ns,
            r.coverage * 100.0,
            cal_str,
            cv_str,
        );
    }
    println!();
    let na = "N/A".to_string();
    println!(
        "Mean nRMSE (headline):         {}",
        mean_nrmse
            .map(|v| format!("{v:.3e}  [lower is better; 0=perfect, >=1=null scheduler]"))
            .unwrap_or(na.clone())
    );
    let cov_label = if zero_coverage_count > 0 {
        format!(
            "Mean coverage:                 {}  [{zero_coverage_count} event(s) with zero coverage]",
            mean_coverage
                .map(|v| format!("{:.1}%", v * 100.0))
                .unwrap_or(na.clone())
        )
    } else {
        format!(
            "Mean coverage:                 {}",
            mean_coverage
                .map(|v| format!("{:.1}%", v * 100.0))
                .unwrap_or(na.clone())
        )
    };
    println!("{cov_label}");
    println!(
        "Mean calibration:              {}",
        mean_calibration
            .map(|v| format!("{:.1}%", v * 100.0))
            .unwrap_or(na)
    );
}

/// Prints evaluation results as a single JSON object to stdout.
#[allow(clippy::too_many_arguments)]
fn print_json(
    ground_truth: &std::path::Path,
    estimated: &std::path::Path,
    bin_ms: u64,
    results: &[EventMetrics],
    mean_nrmse: Option<f64>,
    zero_coverage_count: usize,
    mean_coverage: Option<f64>,
    mean_calibration: Option<f64>,
) {
    let mut per_event_parts: Vec<String> = Vec::new();
    for r in results {
        let nrmse_str = match r.nrmse {
            Some(v) => f64_to_json(v),
            None => "null".to_string(),
        };
        let cal_str = match r.calibration {
            Some(v) => f64_to_json(v),
            None => "null".to_string(),
        };
        let cv_str = match r.gt_cv {
            Some(v) => f64_to_json(v),
            None => "null".to_string(),
        };
        per_event_parts.push(format!(
            "    {{\"event\": {:?}, \"tid\": {}, \"nrmse\": {}, \"coverage\": {}, \"mean_gt_rate_events_per_ns\": {}, \"calibration\": {}, \"gt_cv\": {}}}",
            r.event_name,
            r.tid,
            nrmse_str,
            f64_to_json(r.coverage),
            f64_to_json(r.mean_gt_rate_events_per_ns),
            cal_str,
            cv_str,
        ));
    }
    let mean_nrmse_str = mean_nrmse
        .map(f64_to_json)
        .unwrap_or_else(|| "null".to_string());
    let cov_str = mean_coverage
        .map(f64_to_json)
        .unwrap_or_else(|| "null".to_string());
    let cal_mean_str = mean_calibration
        .map(f64_to_json)
        .unwrap_or_else(|| "null".to_string());
    println!(
        "{{\n  \"ground_truth\": {:?},\n  \"estimated\": {:?},\n  \"bin_width_ms\": {},\n  \"per_event\": [\n{}\n  ],\n  \"mean_nrmse\": {},\n  \"events_with_zero_coverage\": {},\n  \"mean_coverage\": {},\n  \"mean_calibration\": {}\n}}",
        ground_truth.display().to_string(),
        estimated.display().to_string(),
        bin_ms,
        per_event_parts.join(",\n"),
        mean_nrmse_str,
        zero_coverage_count,
        cov_str,
        cal_mean_str,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_avg_averages_within_bin() {
        let points = vec![(0u64, 1.0f64), (50_000_000, 3.0)];
        let bins = bin_avg(&points, 100_000_000);
        assert_eq!(bins.len(), 1);
        assert!((bins[&0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bin_avg_separates_bins() {
        let points = vec![(50_000_000u64, 1.0f64), (150_000_000, 5.0)];
        let bins = bin_avg(&points, 100_000_000);
        assert_eq!(bins.len(), 2);
        assert!((bins[&0] - 1.0).abs() < 1e-12);
        assert!((bins[&1] - 5.0).abs() < 1e-12);
    }

    fn single_series(rate: f64, bins: &[u64], bin_width_ns: u64) -> Vec<(u64, f64)> {
        bins.iter()
            .map(|&b| (b * bin_width_ns + bin_width_ns / 2, rate))
            .collect()
    }

    #[test]
    fn coverage_full_when_all_bins_covered() {
        let bw = 100_000_000u64;
        let gt_pts = single_series(1.0, &[0, 1, 2], bw);
        let est_pts = single_series(1.0, &[0, 1, 2], bw);
        let gt_bins = bin_avg(&gt_pts, bw);
        let est_bins = bin_avg(&est_pts, bw);
        let n_gt = gt_bins.len();
        let covered = gt_bins.keys().filter(|b| est_bins.contains_key(b)).count();
        assert_eq!(covered as f64 / n_gt as f64, 1.0);
    }

    #[test]
    fn coverage_zero_nrmse_is_none_when_no_est_bins() {
        // With no estimated bins at all, nrmse is None (estimator never observed this event).
        // Pre-first-observation bins are excluded; when there are NO estimated bins, the
        // scored window is empty and nrmse is undefined.
        let bw = 100_000_000u64;
        let gt_pts = single_series(1.0, &[0, 1, 2], bw);
        let gt_bins = bin_avg(&gt_pts, bw);
        let est_bins: HashMap<u64, f64> = HashMap::new();
        let mean_gt = gt_bins.values().sum::<f64>() / gt_bins.len() as f64;

        // No first_est_bin → nrmse = None.
        let first_est_bin: Option<u64> = est_bins.keys().copied().min();
        assert!(first_est_bin.is_none(), "est_bins should be empty");
        // Verify the scoring loop would produce None.
        let nrmse: Option<f64> = if mean_gt > 0.0 {
            first_est_bin.and_then(|first_bin| {
                let mut sq_sum = 0.0f64;
                let mut n = 0usize;
                for (&bin, &gt_rate) in &gt_bins {
                    if bin < first_bin {
                        continue;
                    }
                    let er = (est_bins.get(&bin).copied().unwrap_or(0.0) - gt_rate) / mean_gt;
                    sq_sum += er * er;
                    n += 1;
                }
                (n > 0).then(|| (sq_sum / n as f64).sqrt())
            })
        } else {
            None
        };
        assert!(
            nrmse.is_none(),
            "nrmse should be None when est_bins is empty"
        );
    }

    #[test]
    fn nrmse_zero_when_exact_match() {
        let bw = 100_000_000u64;
        let pts = single_series(2.5e-6, &[0, 1, 2], bw);
        let gt_bins = bin_avg(&pts, bw);
        let est_bins = bin_avg(&pts, bw);
        let mean_gt = gt_bins.values().sum::<f64>() / gt_bins.len() as f64;
        let nrmse_sq_sum: f64 = gt_bins
            .iter()
            .map(|(b, &gt_rate)| {
                let est_rate = est_bins.get(b).copied().unwrap_or(0.0);
                let rel_err = (est_rate - gt_rate) / mean_gt;
                rel_err * rel_err
            })
            .sum();
        let nrmse = (nrmse_sq_sum / gt_bins.len() as f64).sqrt();
        assert!(nrmse < 1e-20);
    }

    #[test]
    fn nrmse_excludes_pre_first_observation_bins() {
        // GT: bins [0, 1, 2] at rate 1.0. Est: covers bins [1, 2] only (first observation = bin 1).
        // Scoring window: bins >= 1, so bins 1 and 2 are scored; bin 0 is excluded.
        // Bin 1: est=1, gt=1 → rel_err = 0. Bin 2: est=1, gt=1 → rel_err = 0.
        // nRMSE = 0 (perfect match within the scored window).
        // Coverage = 2/3 (bins 1,2 covered out of 3 GT bins).
        let bw = 100_000_000u64;
        let gt_pts = single_series(1.0, &[0, 1, 2], bw);
        let est_pts = single_series(1.0, &[1, 2], bw);
        let gt_bins = bin_avg(&gt_pts, bw);
        let est_bins = bin_avg(&est_pts, bw);
        let mean_gt = gt_bins.values().sum::<f64>() / gt_bins.len() as f64;
        let first_est_bin = est_bins.keys().copied().min().unwrap();
        assert_eq!(first_est_bin, 1, "first estimated bin should be 1");

        let mut sq_sum = 0.0f64;
        let mut n_scored = 0usize;
        let mut covered = 0usize;
        for (&bin, &gt_rate) in &gt_bins {
            if est_bins.contains_key(&bin) {
                covered += 1;
            }
            if bin < first_est_bin {
                continue;
            }
            let est_rate = est_bins.get(&bin).copied().unwrap_or(0.0);
            let rel_err = (est_rate - gt_rate) / mean_gt;
            sq_sum += rel_err * rel_err;
            n_scored += 1;
        }
        let nrmse = (sq_sum / n_scored as f64).sqrt();
        let coverage = covered as f64 / gt_bins.len() as f64;

        assert!(
            nrmse < 1e-12,
            "nrmse should be 0 for exact match within scored window"
        );
        assert!((coverage - 2.0 / 3.0).abs() < 1e-12, "coverage=2/3");
    }

    #[test]
    fn nrmse_penalizes_uncovered_bins_within_scored_window() {
        // GT: bins [0, 1, 2] at rate 1.0. Est: covers bins [1] only (first observation = bin 1).
        // Scoring window: bins >= 1 → bins 1 and 2 are scored.
        // Bin 1: est=1, gt=1 → rel_err = 0. Bin 2: est=0 (missing), gt=1 → rel_err = -1.
        // nRMSE over 2 scored bins = sqrt((0 + 1) / 2) = sqrt(0.5).
        let bw = 100_000_000u64;
        let gt_pts = single_series(1.0, &[0, 1, 2], bw);
        let est_pts = single_series(1.0, &[1], bw);
        let gt_bins = bin_avg(&gt_pts, bw);
        let est_bins = bin_avg(&est_pts, bw);
        let mean_gt = gt_bins.values().sum::<f64>() / gt_bins.len() as f64;
        let first_est_bin = est_bins.keys().copied().min().unwrap();

        let mut sq_sum = 0.0f64;
        let mut n_scored = 0usize;
        for (&bin, &gt_rate) in &gt_bins {
            if bin < first_est_bin {
                continue;
            }
            let est_rate = est_bins.get(&bin).copied().unwrap_or(0.0);
            let rel_err = (est_rate - gt_rate) / mean_gt;
            sq_sum += rel_err * rel_err;
            n_scored += 1;
        }
        let nrmse = (sq_sum / n_scored as f64).sqrt();
        assert!((nrmse - 0.5f64.sqrt()).abs() < 1e-10, "nrmse={nrmse}");
    }

    #[test]
    fn nrmse_dimensionless_across_magnitudes() {
        // Events at very different rates but same 2x overestimate → same nRMSE.
        let bw = 100_000_000u64;
        let mut nrmses = Vec::new();
        for &rate in &[1.0f64, 1e-4] {
            let gt_pts = single_series(rate, &[0, 1, 2], bw);
            let est_pts = single_series(rate * 2.0, &[0, 1, 2], bw);
            let gt_bins = bin_avg(&gt_pts, bw);
            let est_bins = bin_avg(&est_pts, bw);
            let mean_gt = gt_bins.values().sum::<f64>() / gt_bins.len() as f64;
            let sq_sum: f64 = gt_bins
                .iter()
                .map(|(b, &gt_rate)| {
                    let est_rate = est_bins.get(b).copied().unwrap_or(0.0);
                    let rel_err = (est_rate - gt_rate) / mean_gt;
                    rel_err * rel_err
                })
                .sum();
            nrmses.push((sq_sum / gt_bins.len() as f64).sqrt());
        }
        // Both should be 1.0 (2x overestimate = rel_err of 1.0 per bin).
        for nrmse in &nrmses {
            assert!((nrmse - 1.0).abs() < 1e-12, "nrmse={nrmse}");
        }
    }

    #[test]
    fn normalize_timestamps_shifts_to_zero() {
        let mut series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        series.insert(0, vec![(1_000_000_000, 1.0), (2_000_000_000, 2.0)]);
        normalize_timestamps(&mut series);
        let pts = &series[&0];
        assert_eq!(pts[0].0, 0);
        assert_eq!(pts[1].0, 1_000_000_000);
    }

    #[test]
    fn normalize_timestamps_noop_when_already_zero() {
        let mut series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        series.insert(0, vec![(0, 1.0), (1_000_000_000, 2.0)]);
        normalize_timestamps(&mut series);
        let pts = &series[&0];
        assert_eq!(pts[0].0, 0);
        assert_eq!(pts[1].0, 1_000_000_000);
    }

    #[test]
    fn normalize_timestamps_aligns_multi_series() {
        let mut series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        series.insert(0, vec![(500_000_000, 1.0), (1_500_000_000, 2.0)]);
        series.insert(1, vec![(1_000_000_000, 3.0), (2_000_000_000, 4.0)]);
        normalize_timestamps(&mut series);
        let pts0 = &series[&0];
        assert_eq!(pts0[0].0, 0);
        assert_eq!(pts0[1].0, 1_000_000_000);
        let pts1 = &series[&1];
        assert_eq!(pts1[0].0, 500_000_000);
        assert_eq!(pts1[1].0, 1_500_000_000);
    }

    #[test]
    fn misaligned_timestamps_produce_nonzero_coverage() {
        let bw = 100_000_000u64;
        let offset = 1_000_000_000_000u64;

        let mut gt_series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        gt_series.insert(
            0,
            vec![
                (offset + bw / 2, 1.0),
                (offset + bw + bw / 2, 1.0),
                (offset + 2 * bw + bw / 2, 1.0),
            ],
        );

        let mut est_series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        est_series.insert(
            0,
            vec![(bw / 2, 1.0), (bw + bw / 2, 1.0), (2 * bw + bw / 2, 1.0)],
        );

        normalize_timestamps(&mut gt_series);
        normalize_timestamps(&mut est_series);

        let gt_bins = bin_avg(&gt_series[&0], bw);
        let est_bins = bin_avg(&est_series[&0], bw);
        let covered = gt_bins.keys().filter(|b| est_bins.contains_key(b)).count();
        assert_eq!(
            covered, 3,
            "all GT bins should be covered after normalization"
        );
    }
}
