# Unseen Expert Games Versus Replay in Riichi Mahjong

This repository contains the analysis-only reproduction package for a study of supervised continuation in four-player riichi mahjong. The study asks whether learning previously unseen expert game records improves playing strength more than replaying old records when both branches consume exactly the same number of decisions and optimizer updates.

## Scope

The package covers:

- one observed 4,000-hanchan pilot comparison;
- a subsequently frozen three-parent replication with 24,000 hanchans;
- a separately frozen parent-baseline supplement with 24,000 hanchans.

Only the three independently initialized replication parents enter the model-level summaries. The pilot was observed before the replication protocol was frozen. The parent supplement was specified only after the replication results were observed.

## Main result

The three direct unseen-minus-replay mean-rank differences were `+0.0247`, `-0.0113`, and `-0.0153`, where lower values favor unseen data. Every whole-wall 95% interval included zero. The cross-parent mean was `-0.0007`, with a descriptive interval of `[-0.0554, +0.0541]` conditional on the shared evaluation walls.

These results do not establish equivalence, show that unseen records are ineffective, or imply a general data-scaling law.

## Exploratory analyses added in October 2026

The updated manuscript, *New Expert Games or a Second Epoch? A Compute-Matched Study of Imitation Learning in Riichi Mahjong*, adds analyses of the same recorded outcomes. No new training or matches are included.

Joint resampling of shared wall groups across the three direct panels gives the same pooled mean, `-0.0007`, with a 95% interval of `[-0.0238, +0.0227]` (10,000 resamples). This interval measures evaluation variability for the fixed trained policies. The original descriptive three-parent t interval, `[-0.0554, +0.0541]`, summarizes the three observed parent-level estimates (df=2), conditional on the shared walls. These intervals answer different questions; neither supplies broad evidence about unseen training runs, and the new interval does not replace the frozen summary.

Other additions cover heterogeneity, approximate minimum detectable effects, a post hoc precision bound, pooled comparisons against parents and Mortal, and development cross-entropy versus playing strength. These are exploratory analyses added after observing the results. The smallest symmetric margin containing the pooled 90% interval is a precision description, not evidence of practical equivalence. Development CE is not recomputed and its uncertainty is not estimated; six related branch observations cannot establish that CE and playing strength are unrelated.

## Reproduction

Python 3.10 or later and NumPy are required; Matplotlib is needed only for figures.

```bash
python -m pip install -r requirements.txt
python reproduce.py --write
python supplementary_analysis.py --write
python make_figures.py figures/
```

The figure script requests Latin Modern Roman; install that font for manuscript typography. If it is unavailable, Matplotlib uses a fallback font and reports warnings; numerical results are unaffected.

The script validates the input checksum and recomputes the reported rank statistics using 5,000 bootstrap resamples of complete wall groups. Computed values must agree with the frozen machine summaries to within `1e-9`.

## Files

- `rank_observations.csv`: simulated challenger ranks and aggregate outcomes with normalized wall indices.
- `manifest.json`: panel relationships, frozen analysis settings, expected statistics, and diagnostic development losses.
- `reproduce.py`: independent rank-statistics reproduction script.
- `analysis_results.json`: reproduced output generated from the package.
- `supplementary_analysis.py`: exploratory pooled analyses using joint whole-wall resampling.
- `supplementary_results.json`: reference output of the exploratory analyses.
- `make_figures.py`: draws manuscript Figures 2 and 3 from the two result files.
- `ARTIFACT.txt`: detailed scope, chronology, and interpretation notes.

## Data and rights

This repository does not contain Tenhou source logs, source game identifiers, player names, replay URLs, complete trajectories, trained policy weights, or third-party Mortal weights. It does not provide a method for downloading Tenhou records.

In correspondence about this study, Tenhou support indicated that records may be used for research, but publication should be limited to analysis results and exclude source game records and trained networks, irrespective of commercial status. This repository does not grant any rights to Tenhou materials. Anyone seeking to use Tenhou records should review the official Tenhou manual and contact `support@c-egg.com` when required.

The original code and accompanying materials in this repository are released under the MIT License (see `LICENSE`). This license does not grant any rights to third-party Tenhou materials.
