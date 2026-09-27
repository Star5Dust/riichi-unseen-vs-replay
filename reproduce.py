"""Recompute fixed rank statistics from anonymous simulated outcomes; no training."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def bootstrap(values, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.concatenate([
        values[rng.integers(len(values), size=(min(100, 5000-i), len(values)))].mean(axis=1)
        for i in range(0, 5000, 100)])
    return np.quantile(estimates, [.025, .975]).tolist()

def cross(values):
    a = np.asarray(values, dtype=np.float64)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    margin = 4.302652729911275 * sd / np.sqrt(3)
    return [mean, sd, mean-margin, mean+margin]

def main(write=False):
    m = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    require(hashlib.sha256((HERE/'rank_observations.csv').read_bytes()).hexdigest()
            == m['rank_csv_sha256'], 'CSV checksum mismatch')
    groups = defaultdict(lambda: defaultdict(dict))
    with (HERE/'rank_observations.csv').open(encoding='utf-8', newline='') as f:
        for raw in csv.DictReader(f):
            panel = raw.pop('panel')
            require(panel in m['panels'], 'Unknown panel')
            row = {k:int(v) for k,v in raw.items()}
            wall, rotation = row['wall_index'], row['rotation']
            require(rotation in range(4) and rotation not in groups[panel][wall], 'Duplicate rotation')
            require(row['challenger_rank'] in range(1,5), 'Invalid rank')
            groups[panel][wall][rotation] = row
    stats, ranks, auxiliary = {}, {}, {}
    for panel, spec in m['panels'].items():
        require(sorted(groups[panel]) == list(range(spec['wall_groups'])), 'Missing wall')
        order = spec['bootstrap_wall_order']
        require(sorted(order) == list(range(spec['wall_groups'])), 'Invalid bootstrap order')
        require(all(set(v)==set(range(4)) for v in groups[panel].values()), 'Missing seat rotation')
        ranks[panel] = {w:sum(groups[panel][w][r]['challenger_rank'] for r in range(4))/4 for w in order}
        deltas = [sum((groups[panel][w][r]['challenger_rank'] -
                      (10-groups[panel][w][r]['challenger_rank'])/3) for r in range(4))/4 for w in order]
        mean = float(np.mean(list(ranks[panel].values())))
        stats[panel] = [mean, (10-mean)/3, float(np.mean(deltas)),
                        *bootstrap(deltas, spec['bootstrap_seed'])]
        rows = [row for wall in groups[panel].values() for row in wall.values()]
        totals = {k:sum(row[k] for row in rows) for k in (
            'challenger_net_points','challenger_wins','challenger_rounds','challenger_dealin_rounds',
            'opponent_net_points_sum','opponent_wins_sum','opponent_rounds_sum','opponent_dealin_rounds_sum')}
        auxiliary[panel] = {'hanchans':len(rows),
            'challenger_net_points':totals['challenger_net_points']/len(rows),
            'challenger_win_rate':totals['challenger_wins']/totals['challenger_rounds'],
            'challenger_dealin_rate':totals['challenger_dealin_rounds']/totals['challenger_rounds'],
            'opponent_net_points_per_seat':totals['opponent_net_points_sum']/(3*len(rows)),
            'opponent_win_rate':totals['opponent_wins_sum']/totals['opponent_rounds_sum'],
            'opponent_dealin_rate':totals['opponent_dealin_rounds_sum']/totals['opponent_rounds_sum']}
    for name, spec in m['paired_comparisons'].items():
        a, b = spec['new_panel'], spec['old_panel']
        require(m['panels'][a]['wall_set']==m['panels'][b]['wall_set'] and
                ranks[a].keys()==ranks[b].keys(), 'Pairing mismatch')
        values = [ranks[a][w]-ranks[b][w] for w in sorted(ranks[a])]
        stats[name] = [float(np.mean(values)), *bootstrap(values, spec['bootstrap_seed'])]
    for name, panels in m['cross_chain'].items():
        require(len(panels)==3, 'Cross-parent sample must be three')
        stats[name] = cross([stats[x][2] for x in panels])
    require(stats.keys()==m['expected_statistics'].keys(), 'Statistic coverage mismatch')
    for name, expected in m['expected_statistics'].items():
        require(np.allclose(stats[name], expected, rtol=0, atol=1e-9),
                name + ': differs from frozen machine summary')
    result = {'statistics_schema':m['statistics_schema'], 'statistics':stats,
              'auxiliary':auxiliary, 'development_ce':m['development_ce'],
              'counts':m['counts'], 'verification':'All statistics match frozen summaries within 1e-9.'}
    if write:
        (HERE/'analysis_results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f"PASS: {len(stats)} rank summaries; {sum(x['hanchans'] for x in auxiliary.values()):,} hanchans; pilot excluded from three-parent summaries.")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)

