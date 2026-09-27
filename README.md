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

## Reproduction

Python 3.10 or later and NumPy are required.

```bash
python -m pip install -r requirements.txt
python reproduce.py --write
```

The script validates the input checksum and recomputes the reported rank statistics using 5,000 bootstrap resamples of complete wall groups. Computed values must agree with the frozen machine summaries to within `1e-9`.

## Files

- `rank_observations.csv`: simulated challenger ranks and aggregate outcomes with normalized wall indices.
- `manifest.json`: panel relationships, frozen analysis settings, expected statistics, and diagnostic development losses.
- `reproduce.py`: independent rank-statistics reproduction script.
- `analysis_results.json`: reproduced output generated from the package.
- `ARTIFACT.txt`: detailed scope, chronology, and interpretation notes.

## Data and rights

This repository does not contain Tenhou source logs, source game identifiers, player names, replay URLs, complete trajectories, trained policy weights, or third-party Mortal weights. It does not provide a method for downloading Tenhou records.

Tenhou states that copyright in its game records belongs to Tenhou and places conditions on uses beyond enjoying Tenhou play. This repository does not grant any rights to Tenhou materials. Anyone seeking to use Tenhou records should review the official Tenhou manual and contact `support@c-egg.com` when required.

The original code and accompanying materials in this repository are released under the MIT License (see `LICENSE`). This license does not grant any rights to third-party Tenhou materials.
