use crate::perfetto;
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

struct EventMetrics {
    event_name: String,
    tid: u32,
    rmse_events_per_ns: Option<f64>,
    coverage: f64,
    mean_gt_rate_events_per_ns: f64,
}

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
    let mut pooled_sq_err_sum = 0.0f64;
    let mut pooled_n = 0usize;

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

        let n_gt = gt_bins.len();
        if n_gt == 0 {
            continue;
        }

        // Per-event RMSE and coverage.
        let mut sq_err_sum = 0.0f64;
        let mut covered = 0usize;
        for (&bin, &gt_rate) in &gt_bins {
            let est_rate = est_bins.get(&bin).copied().unwrap_or(0.0);
            // Pooled RMSE uses est=0 for uncovered bins.
            pooled_sq_err_sum += (est_rate - gt_rate).powi(2);
            pooled_n += 1;
            if let Some(&er) = est_bins.get(&bin) {
                sq_err_sum += (er - gt_rate).powi(2);
                covered += 1;
            }
        }

        let coverage = covered as f64 / n_gt as f64;
        let per_event_rmse = if covered > 0 {
            Some((sq_err_sum / covered as f64).sqrt())
        } else {
            None
        };
        let mean_gt = gt_bins.values().sum::<f64>() / n_gt as f64;

        results.push(EventMetrics {
            event_name: event_name.clone(),
            tid: *tid,
            rmse_events_per_ns: per_event_rmse,
            coverage,
            mean_gt_rate_events_per_ns: mean_gt,
        });
    }

    let pooled_rmse = if pooled_n > 0 {
        Some((pooled_sq_err_sum / pooled_n as f64).sqrt())
    } else {
        None
    };

    let with_rmse: Vec<f64> = results
        .iter()
        .filter_map(|r| r.rmse_events_per_ns)
        .collect();
    let macro_avg_rmse = if with_rmse.is_empty() {
        None
    } else {
        Some(with_rmse.iter().sum::<f64>() / with_rmse.len() as f64)
    };
    let zero_coverage_count = results.iter().filter(|r| r.rmse_events_per_ns.is_none()).count();
    let mean_coverage = if results.is_empty() {
        None
    } else {
        Some(results.iter().map(|r| r.coverage).sum::<f64>() / results.len() as f64)
    };

    if json {
        print_json(
            &ground_truth,
            &estimated,
            bin_ms,
            &results,
            pooled_rmse,
            macro_avg_rmse,
            zero_coverage_count,
            mean_coverage,
        );
    } else {
        print_text(
            &ground_truth,
            &estimated,
            bin_ms,
            &results,
            pooled_rmse,
            macro_avg_rmse,
            zero_coverage_count,
            mean_coverage,
        );
    }

    Ok(())
}

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

fn bin_avg(points: &[(u64, f64)], bin_width_ns: u64) -> HashMap<u64, f64> {
    let mut acc: HashMap<u64, (f64, usize)> = HashMap::new();
    for &(ts, rate) in points {
        let bin = ts / bin_width_ns;
        let e = acc.entry(bin).or_default();
        e.0 += rate;
        e.1 += 1;
    }
    acc.into_iter().map(|(b, (s, n))| (b, s / n as f64)).collect()
}

#[allow(clippy::too_many_arguments)]
fn print_text(
    ground_truth: &std::path::Path,
    estimated: &std::path::Path,
    bin_ms: u64,
    results: &[EventMetrics],
    pooled_rmse: Option<f64>,
    macro_avg_rmse: Option<f64>,
    zero_coverage_count: usize,
    mean_coverage: Option<f64>,
) {
    println!("Evaluation Report");
    println!("=================");
    println!("Ground truth: {}", ground_truth.display());
    println!("Estimated:    {}", estimated.display());
    println!("Bin width:    {bin_ms} ms");
    println!();
    println!(
        "{:<32} {:>5}  {:<16}  Coverage",
        "Event", "TID", "RMSE (ev/ns)"
    );
    println!("{}", "-".repeat(66));
    for r in results {
        let rmse_str = match r.rmse_events_per_ns {
            Some(v) => format!("{v:.3e}"),
            None => "—".to_string(),
        };
        println!(
            "{:<32} {:>5}  {:<16}  {:.1}%",
            r.event_name,
            r.tid,
            rmse_str,
            r.coverage * 100.0
        );
    }
    println!();
    let na = "N/A".to_string();
    println!(
        "Pooled RMSE (est=0 for gaps):  {}",
        pooled_rmse.map(|v| format!("{v:.3e} ev/ns")).unwrap_or(na.clone())
    );
    let macro_label = if zero_coverage_count > 0 {
        format!(
            "Macro-avg RMSE ({}/{} events): {} ev/ns  [{zero_coverage_count} event(s) excluded: no coverage]",
            results.len() - zero_coverage_count,
            results.len(),
            macro_avg_rmse.map(|v| format!("{v:.3e}")).unwrap_or(na.clone())
        )
    } else {
        format!(
            "Macro-avg RMSE:                {}",
            macro_avg_rmse.map(|v| format!("{v:.3e} ev/ns")).unwrap_or(na.clone())
        )
    };
    println!("{macro_label}");
    println!(
        "Mean coverage:                 {}",
        mean_coverage.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or(na)
    );
}

#[allow(clippy::too_many_arguments)]
fn print_json(
    ground_truth: &std::path::Path,
    estimated: &std::path::Path,
    bin_ms: u64,
    results: &[EventMetrics],
    pooled_rmse: Option<f64>,
    macro_avg_rmse: Option<f64>,
    zero_coverage_count: usize,
    mean_coverage: Option<f64>,
) {
    let mut per_event_parts: Vec<String> = Vec::new();
    for r in results {
        let rmse_str = match r.rmse_events_per_ns {
            Some(v) => format!("{v}"),
            None => "null".to_string(),
        };
        per_event_parts.push(format!(
            "    {{\"event\": {:?}, \"tid\": {}, \"rmse_events_per_ns\": {}, \"coverage\": {}, \"mean_gt_rate_events_per_ns\": {}}}",
            r.event_name,
            r.tid,
            rmse_str,
            r.coverage,
            r.mean_gt_rate_events_per_ns
        ));
    }
    let pooled_str = pooled_rmse.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
    let macro_str = macro_avg_rmse.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
    let cov_str = mean_coverage.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
    println!(
        "{{\n  \"ground_truth\": {:?},\n  \"estimated\": {:?},\n  \"bin_width_ms\": {},\n  \"per_event\": [\n{}\n  ],\n  \"pooled_rmse_events_per_ns\": {},\n  \"macro_avg_rmse_events_per_ns\": {},\n  \"events_with_zero_coverage\": {},\n  \"mean_coverage\": {}\n}}",
        ground_truth.display().to_string(),
        estimated.display().to_string(),
        bin_ms,
        per_event_parts.join(",\n"),
        pooled_str,
        macro_str,
        zero_coverage_count,
        cov_str,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_avg_averages_within_bin() {
        // Two points at t=0 and t=50ms both fall in bin 0 (bin_width=100ms).
        let points = vec![(0u64, 1.0f64), (50_000_000, 3.0)];
        let bins = bin_avg(&points, 100_000_000);
        assert_eq!(bins.len(), 1);
        assert!((bins[&0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bin_avg_separates_bins() {
        // t=50ms → bin 0, t=150ms → bin 1 (bin_width=100ms).
        let points = vec![(50_000_000u64, 1.0f64), (150_000_000, 5.0)];
        let bins = bin_avg(&points, 100_000_000);
        assert_eq!(bins.len(), 2);
        assert!((bins[&0] - 1.0).abs() < 1e-12);
        assert!((bins[&1] - 5.0).abs() < 1e-12);
    }

    fn single_series(rate: f64, bins: &[u64], bin_width_ns: u64) -> Vec<(u64, f64)> {
        bins.iter().map(|&b| (b * bin_width_ns + bin_width_ns / 2, rate)).collect()
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
    fn coverage_zero_when_no_estimates() {
        let bw = 100_000_000u64;
        let gt_pts = single_series(1.0, &[0, 1, 2], bw);
        let gt_bins = bin_avg(&gt_pts, bw);
        let est_bins: HashMap<u64, f64> = HashMap::new();
        let covered = gt_bins.keys().filter(|b| est_bins.contains_key(b)).count();
        assert_eq!(covered, 0);
        // With no covered bins, RMSE should be None.
        let per_event_rmse: Option<f64> = if covered > 0 { Some(0.0) } else { None };
        assert!(per_event_rmse.is_none());
    }

    #[test]
    fn rmse_zero_when_exact_match() {
        let bw = 100_000_000u64;
        let pts = single_series(2.5e-6, &[0, 1, 2], bw);
        let gt_bins = bin_avg(&pts, bw);
        let est_bins = bin_avg(&pts, bw);
        let mut sq_err = 0.0f64;
        let mut n = 0usize;
        for (b, gt_rate) in &gt_bins {
            if let Some(&er) = est_bins.get(b) {
                sq_err += (er - gt_rate).powi(2);
                n += 1;
            }
        }
        let rmse = (sq_err / n as f64).sqrt();
        assert!(rmse < 1e-20);
    }

    #[test]
    fn pooled_rmse_penalizes_uncovered_bins() {
        let bw = 100_000_000u64;
        // GT has 2 bins with rate 1.0; estimated has only bin 0.
        let gt_pts = single_series(1.0, &[0, 1], bw);
        let est_pts = single_series(1.0, &[0], bw);
        let gt_bins = bin_avg(&gt_pts, bw);
        let est_bins = bin_avg(&est_pts, bw);
        let mut sq_err = 0.0f64;
        let mut n = 0usize;
        for (b, &gt_rate) in &gt_bins {
            let er = est_bins.get(b).copied().unwrap_or(0.0);
            sq_err += (er - gt_rate).powi(2);
            n += 1;
        }
        let pooled_rmse = (sq_err / n as f64).sqrt();
        // bin 0: err=0, bin 1: err=1.0^2 → pooled_rmse = sqrt(0.5) ≈ 0.707
        assert!((pooled_rmse - 0.5f64.sqrt()).abs() < 1e-10);
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
        // global min is 500_000_000; both series shifted by that amount
        let pts0 = &series[&0];
        assert_eq!(pts0[0].0, 0);
        assert_eq!(pts0[1].0, 1_000_000_000);
        let pts1 = &series[&1];
        assert_eq!(pts1[0].0, 500_000_000);
        assert_eq!(pts1[1].0, 1_500_000_000);
    }

    #[test]
    fn misaligned_timestamps_produce_nonzero_coverage() {
        // Simulate the bug: GT has absolute timestamps at ~10^12 ns offset,
        // estimated starts near 0. Without normalization coverage would be 0.
        let bw = 100_000_000u64; // 100 ms bins
        let offset = 1_000_000_000_000u64; // ~10^12 ns

        // GT: bins 0,1,2 at the offset origin
        let mut gt_series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        gt_series.insert(0, vec![
            (offset + bw / 2,       1.0),
            (offset + bw + bw / 2,  1.0),
            (offset + 2 * bw + bw / 2, 1.0),
        ]);

        // Estimated: bins 0,1,2 starting near 0
        let mut est_series: HashMap<u32, Vec<(u64, f64)>> = HashMap::new();
        est_series.insert(0, vec![
            (bw / 2,       1.0),
            (bw + bw / 2,  1.0),
            (2 * bw + bw / 2, 1.0),
        ]);

        normalize_timestamps(&mut gt_series);
        normalize_timestamps(&mut est_series);

        let gt_bins = bin_avg(&gt_series[&0], bw);
        let est_bins = bin_avg(&est_series[&0], bw);
        let covered = gt_bins.keys().filter(|b| est_bins.contains_key(b)).count();
        assert_eq!(covered, 3, "all GT bins should be covered after normalization");
    }
}
