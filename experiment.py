"""Auditable static 2D localization experiment. No model credentials required.

Offline controller is a feedback heuristic, NOT an LLM run.
Run: python experiment.py --output results
Each proposal is evaluated on the SAME development cases. Test truth is used
only after selection, never by the controller. Units: metres.
"""
import argparse
import itertools
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import least_squares

SCENARIOS = [("clean", 0.0, False), ("outliers", 0.25, False),
             ("poor_geometry", 0.25, True)]

def cases(seed, count=30):
    rng = np.random.default_rng(seed)
    data = []
    for label, contamination, narrow in SCENARIOS:
        for _ in range(count):
            truth = rng.uniform(-10, 10, 2)
            angles = rng.normal(0, 0.08, 12) if narrow else rng.uniform(0, 2*np.pi, 12)
            radius = rng.uniform(15, 45, 12)
            anchors = truth + np.column_stack((np.cos(angles), np.sin(angles)))*radius[:, None]
            observed_anchors = anchors + rng.normal(0, 0.5, anchors.shape)
            distances = radius + rng.normal(0, 1, 12)
            k = round(contamination * len(radius))
            if k:
                idx = rng.choice(len(radius), k, replace=False)
                distances[idx] += rng.uniform(15, 30, k)
            data.append(dict(scenario=label, anchors=observed_anchors,
                             distances=np.maximum(0.1, distances), truth=truth))
    return data

def locate(case, config):
    anchors, distances = case['anchors'], case['distances']
    residual = lambda x: np.linalg.norm(x-anchors, axis=1)-distances
    center = anchors.mean(axis=0)
    # All methods use the same four initial guesses and the same per-start cap.
    starts = [center, center+[20, 0], center+[-20, 0], center+[0, 20]]
    solutions = [least_squares(residual, x, loss=config['loss'],
                  f_scale=config['scale'], max_nfev=100) for x in starts]
    result = min(solutions, key=lambda s: s.cost)
    r = residual(result.x)
    return result.x, {'median_abs_residual': float(np.median(np.abs(r))),
                     'nfev': sum(s.nfev for s in solutions),
                     'all_starts_failed': not any(s.success for s in solutions)}

def evaluate(data, config):
    t = time.perf_counter()
    errors = {s[0]: [] for s in SCENARIOS}
    medians, nfev, failed = [], 0, 0
    for case in data:
        prediction, diagnostics = locate(case, config)
        errors[case['scenario']].append(float(np.linalg.norm(prediction-case['truth'])))
        medians.append(diagnostics['median_abs_residual'])
        nfev += diagnostics['nfev']
        failed += int(diagnostics['all_starts_failed'])
    return {'by_scenario': {key: {'median_m': float(np.median(v)),
                'cep90_m': float(np.quantile(v, 0.9)), 'count': len(v)}
            for key, v in errors.items()},
            'mean_cep90_m': float(np.mean([np.quantile(v, 0.9) for v in errors.values()])),
            'median_abs_residual_m': float(np.median(medians)),
            'seconds': time.perf_counter()-t, 'nfev': nfev,
            'all_starts_failed_cases': failed, 'errors_m': errors}

def propose(history, pool):
    used = {(h['config']['loss'], h['config']['scale']) for h in history}
    available = [c for c in pool if (c['loss'], c['scale']) not in used]
    if not history:
        return {'loss': 'linear', 'scale': 1.0}, '建立平方损失基线'
    if len(history) == 1:
        return {'loss': 'huber', 'scale': 3.0}, '基线反馈后尝试减轻大残差影响'
    best = min(history, key=lambda h: h['selection_score'])
    target = 'cauchy' if best['config']['loss'] != 'cauchy' else 'soft_l1'
    reason = '依据当前最佳损失探索备选损失，并用残差反馈选择尺度'
    preferred = [c for c in available if c['loss'] == target] or available
    # Match the scale to residual feedback without reading any individual truth.
    residual_scale = max(1.0, 2*best['metrics']['median_abs_residual_m'])
    choice = min(preferred, key=lambda c: abs(c['scale']-residual_scale))
    return choice, reason

def run(output, seed=20261010, count=30, rounds=6):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    development, test = cases(seed, count), cases(seed+1000, count)
    pool = [dict(loss=l, scale=float(s)) for l, s in
            itertools.product(['linear', 'huber', 'soft_l1', 'cauchy'], [1, 3, 8])]
    baseline_dev = evaluate(development, {'loss': 'linear', 'scale': 1.0})
    clean_limit = baseline_dev['by_scenario']['clean']['cep90_m']*1.10
    history = []
    for i in range(rounds):
        config, reason = propose(history, pool)
        metrics = baseline_dev if i == 0 else evaluate(development, config)
        clean_excess = max(0, metrics['by_scenario']['clean']['cep90_m']-clean_limit)
        score = metrics['mean_cep90_m']+10*clean_excess
        record = dict(round=i+1, controller='offline_feedback_heuristic',
                      config=config, reason=reason, selection_score=score,
                      clean_constraint_satisfied=clean_excess == 0, metrics=metrics)
        history.append(record)
        print(i+1, config, round(score, 3))
    selected = min(history, key=lambda h: h['selection_score'])
    rng = np.random.default_rng(seed+2000)
    # Equal number of candidate evaluations, same solver and development cases.
    random_configs = [pool[i] for i in rng.choice(len(pool), rounds, replace=False)]
    random_records = []
    for config in random_configs:
        metrics = evaluate(development, config)
        penalty = 10*max(0, metrics['by_scenario']['clean']['cep90_m']-clean_limit)
        random_records.append(dict(config=config, score=metrics['mean_cep90_m']+penalty,
                                   metrics=metrics))
    random_best = min(random_records, key=lambda h: h['score'])
    test_methods = {'least_squares': {'loss': 'linear', 'scale': 1.0},
                    'fixed_huber': {'loss': 'huber', 'scale': 3.0},
                    'feedback_selected': selected['config'],
                    'random_selected': random_best['config']}
    test_results = {name: evaluate(test, config) for name, config in test_methods.items()}
    summary = dict(seed=seed, cases_per_scenario_per_split=count, rounds=rounds,
                   units='metres', status='offline prototype; no LLM calls',
                   scope='static synthetic data; no RSSI calibration or mobile target',
                   development_seed=seed, test_seed=seed+1000,
                   clean_dev_limit_m=clean_limit, selected=selected['config'],
                   random_selected=random_best['config'], test=test_results,
                   environment={'python': platform.python_version(),
                                'numpy': np.__version__, 'scipy': scipy.__version__})
    (output/'iterations.jsonl').write_text('\n'.join(json.dumps(h, ensure_ascii=False) for h in history)+'\n')
    (output/'random_search.json').write_text(json.dumps(random_records, indent=2))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    serialized = [{**c, 'anchors': c['anchors'].tolist(), 'distances': c['distances'].tolist(),
                   'truth': c['truth'].tolist()} for c in test]
    (output/'test_cases.json').write_text(json.dumps(serialized))
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    if args.count < 2 or not 1 <= args.rounds <= 12:
        parser.error('count >= 2 and rounds in [1, 12] are required')
    run(args.output, args.seed, args.count, args.rounds)
