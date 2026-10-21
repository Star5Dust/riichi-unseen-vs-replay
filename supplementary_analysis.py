"""Additional analyses reported in the TMLR manuscript.

All quantities are computed from the already-recorded match outcomes in
rank_observations.csv and the development cross-entropy values in
manifest.json. No game is simulated and no model is trained or evaluated.

Usage:
    python supplementary_analysis.py            # print results
    python supplementary_analysis.py --write    # also write supplementary_results.json

Conventions
-----------
For one hanchan in which a single challenger sits with three copies of a
reference policy, the per-game rank gap is
    gap = challenger_rank - mean_rank_of_the_three_reference_seats
        = (4 * challenger_rank - 10) / 3,
because the four ranks always sum to 10. Lower values favour the challenger.
A gap of 0 is the value expected if the challenger and the reference policy
are equally strong. Each wall group contributes the mean over its four seat
rotations; all resampling is over complete wall groups.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = ["20260921", "20260922", "20260923"]
N_BOOT = 10_000
RNG_SEED = 20261020


def load():
    ranks = defaultdict(dict)
    with (HERE / "rank_observations.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ranks[row["panel"]][(int(row["wall_index"]), int(row["rotation"]))] = int(row["challenger_rank"])
    out = {}
    for panel, d in ranks.items():
        n_wall = max(w for w, _ in d) + 1
        a = np.zeros((n_wall, 4))
        for (w, r), v in d.items():
            a[w, r] = v
        out[panel] = a
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    return out, manifest


def gap_per_wall(rank_matrix):
    return ((4.0 * rank_matrix - 10.0) / 3.0).mean(axis=1)


def joint_bootstrap(per_wall_list, stat, rng):
    """Resample wall groups jointly for panels that share the same walls."""
    n = per_wall_list[0].shape[0]
    assert all(x.shape[0] == n for x in per_wall_list)
    stacked = np.stack(per_wall_list)  # panels x walls
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return np.array([stat(stacked[:, i]) for i in idx])


def ci(samples, level=0.95):
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return [float(lo), float(hi)]


def main(write):
    rng = np.random.default_rng(RNG_SEED)
    R, m = load()
    res = {}

    # 1. Pooled same-table comparison across the three parents (shared walls).
    direct = [gap_per_wall(R[f"replication/{s}/direct"]) for s in SEEDS]
    pooled = float(np.mean([d.mean() for d in direct]))
    boot = joint_bootstrap(direct, lambda x: x.mean(), rng)
    per_panel_se = [float(joint_bootstrap([d], lambda x: x.mean(), rng).std()) for d in direct]
    res["direct_pooled"] = {
        "estimate": pooled,
        "ci95": ci(boot, 0.95),
        "ci90": ci(boot, 0.90),
        "bootstrap_se": float(boot.std()),
        "per_panel_bootstrap_se": per_panel_se,
        "per_panel_mde_80pct_power": [2.8 * s for s in per_panel_se],
        "pooled_mde_80pct_power": 2.8 * float(boot.std()),
        "note": "Conditional on the trained policies; quantifies evaluation variability across sampled wall groups, not variability across new training runs.",
    }

    # 2. Heterogeneity across parents: are the three direct gaps compatible with one common value?
    #    Use the joint-bootstrap covariance, which accounts for shared walls.
    boot_vec = joint_bootstrap(direct, lambda x: x.mean(axis=1), rng)  # N_BOOT x 3
    cov = np.cov(boot_vec.T)
    est = np.array([d.mean() for d in direct])
    C = np.array([[1, -1, 0], [0, 1, -1]], dtype=float)
    diff = C @ est
    q = float(diff @ np.linalg.solve(C @ cov @ C.T, diff))
    from math import exp
    p_value = exp(-q / 2)  # chi-square with 2 df survival function
    res["direct_heterogeneity"] = {
        "per_parent_gaps": est.tolist(),
        "between_parent_sd": float(est.std(ddof=1)),
        "mean_within_panel_se": float(np.mean(per_panel_se)),
        "wald_q_2df": q,
        "p_value": p_value,
        "cross_panel_correlation": np.corrcoef(boot_vec.T).tolist(),
    }

    # 3. Exploratory equivalence bound: smallest symmetric margin containing the 90% interval.
    lo90, hi90 = res["direct_pooled"]["ci90"]
    res["direct_equivalence_bound_90"] = float(max(abs(lo90), abs(hi90)))

    # 4. Strength relative to Mortal (absolute, not only differences).
    mortal = {(s, b): R[f"replication/{s}/mortal/{b}"] for s in SEEDS for b in ("new", "old")}
    allm = np.concatenate([v.ravel() for v in mortal.values()])
    mort_walls = [gap_per_wall(v) for v in mortal.values()]
    boot_m = joint_bootstrap(mort_walls, lambda x: x.mean(), rng)
    res["vs_mortal"] = {
        "per_branch_mean_rank": {f"{s}/{'unseen' if b == 'new' else 'replay'}": float(v.mean()) for (s, b), v in mortal.items()},
        "pooled_mean_rank": float(allm.mean()),
        "pooled_gap": float(np.mean([w.mean() for w in mort_walls])),
        "pooled_gap_ci95": ci(boot_m),
        "pooled_rank_distribution": [float((allm == k).mean()) for k in (1, 2, 3, 4)],
        "hanchans": int(allm.size),
    }
    paired = [gap_per_wall(mortal[(s, "new")]) - gap_per_wall(mortal[(s, "old")]) for s in SEEDS]
    # report on the challenger-rank scale used in the original package (mean rank difference)
    paired_rank = [mortal[(s, "new")].mean(axis=1) - mortal[(s, "old")].mean(axis=1) for s in SEEDS]
    boot_pr = joint_bootstrap(paired_rank, lambda x: x.mean(), rng)
    res["vs_mortal_paired_unseen_minus_replay_rank_pooled"] = {
        "estimate": float(np.mean([p.mean() for p in paired_rank])),
        "ci95": ci(boot_pr),
    }

    # 5. Continuation versus own parent (supplement; all six panels share walls).
    sup = {(s, b): gap_per_wall(R[f"supplement/{s}/{b}"]) for s in SEEDS for b in ("new", "old")}
    for b, name in (("new", "unseen"), ("old", "replay")):
        arrs = [sup[(s, b)] for s in SEEDS]
        bb = joint_bootstrap(arrs, lambda x: x.mean(), rng)
        res[f"{name}_vs_parent_pooled"] = {"estimate": float(np.mean([a.mean() for a in arrs])), "ci95": ci(bb)}
    allc = [sup[k] for k in sup]
    bb = joint_bootstrap(allc, lambda x: x.mean(), rng)
    res["any_continuation_vs_parent_pooled"] = {"estimate": float(np.mean([a.mean() for a in allc])), "ci95": ci(bb)}

    # 6. Rank distributions in the same-table panels (challenger = unseen branch).
    res["direct_rank_distribution_unseen"] = {
        s: [float((R[f"replication/{s}/direct"] == k).mean()) for k in (1, 2, 3, 4)] for s in SEEDS
    }

    # 7. Development cross-entropy versus playing strength relative to the parent.
    ce = m["development_ce"]
    rows = []
    for s in SEEDS:
        for b, name in (("new", "unseen"), ("old", "replay")):
            rows.append({
                "parent": s, "branch": name,
                "ce_parent": ce[f"{s}/100k"], "ce_branch": ce[f"{s}/{b}"],
                "delta_ce": ce[f"{s}/{b}"] - ce[f"{s}/100k"],
                "relative_delta_ce": (ce[f"{s}/{b}"] - ce[f"{s}/100k"]) / ce[f"{s}/100k"],
                "gap_vs_parent": float(sup[(s, b)].mean()),
            })
    dce = np.array([r["delta_ce"] for r in rows])
    gp = np.array([r["gap_vs_parent"] for r in rows])
    res["ce_vs_play"] = {
        "rows": rows,
        "mean_delta_ce": float(dce.mean()),
        "mean_relative_delta_ce": float(np.mean([r["relative_delta_ce"] for r in rows])),
        "pearson_r_delta_ce_vs_gap": float(np.corrcoef(dce, gp)[0, 1]),
        "unseen_minus_replay_ce": {s: ce[f"{s}/new"] - ce[f"{s}/old"] for s in SEEDS},
    }

    print(json.dumps(res, indent=2))
    if write:
        (HERE / "supplementary_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
