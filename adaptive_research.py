"""Follow-up experiment: observation-only gating and noise-adaptive trimming.

The current Codex conversation proposes hypotheses and reads tool feedback.
This script is a numerical tool, not a standalone model controller.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from experiment import SCENARIOS, cases, locate


def dataset(seed, count, noise):
    data = cases(seed, count)
    if noise > 1:
        rng = np.random.default_rng(seed + 700000)
        for case in data:
            case['distances'] = np.maximum(.1, case['distances'] +
                rng.normal(0, np.sqrt(noise**2 - 1), len(case['distances'])))
    return data


def adaptive_locate(case, config):
    ls, info = locate(case, {'loss': 'linear', 'scale': 1.})
    residual = lambda x: np.linalg.norm(x-case['anchors'], axis=1)-case['distances']
    if np.sqrt(np.mean(residual(ls)**2)) <= config['gate']:
        return ls, {**info, 'route': 'least_squares', 'removed': 0}
    robust, extra = locate(case, {'loss': 'cauchy', 'scale': config['scale']})
    info['nfev'] += extra['nfev']
    r = residual(robust)
    sigma = 1.4826 * np.median(np.abs(r-np.median(r)))
    threshold = max(config['trim'], config.get('mad_multiplier', 0.) * sigma)
    mask = np.abs(r) <= threshold
    # Need at least four observations; rank alone does not certify identifiability.
    if np.count_nonzero(mask) < 4:
        return robust, {**info, 'route': 'robust_fallback', 'removed': 0}
    filtered = {'anchors': case['anchors'][mask], 'distances': case['distances'][mask]}
    refit, extra = locate(filtered, {'loss': 'linear', 'scale': 1.})
    info['nfev'] += extra['nfev']
    info['all_starts_failed'] = extra['all_starts_failed']
    return refit, {**info, 'route': 'trim_refit', 'removed': int((~mask).sum())}


def evaluate(data, config):
    errors = {s[0]: [] for s in SCENARIOS}
    routes, removed, nfev = {}, 0, 0
    start = time.perf_counter()
    for case in data:
        if config['method'] == 'adaptive':
            prediction, diagnostic = adaptive_locate(case, config)
        else:
            prediction, diagnostic = locate(case, config)
        errors[case['scenario']].append(float(np.linalg.norm(prediction-case['truth'])))
        route = diagnostic.get('route', config['method'])
        routes[route] = routes.get(route, 0) + 1
        removed += diagnostic.get('removed', 0)
        nfev += diagnostic['nfev']
    return {'cep90_m': {s: float(np.quantile(e, .9)) for s, e in errors.items()},
            'errors_m': errors, 'routes': routes, 'removed': removed,
            'nfev': nfev, 'seconds': time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['development', 'test'], required=True)
    parser.add_argument('--output', default='results/adaptive')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'development':
        configs = {
            'least_squares': {'method': 'fixed', 'loss': 'linear', 'scale': 1.},
            'fixed_cauchy': {'method': 'fixed', 'loss': 'cauchy', 'scale': 3.},
            'fixed_trim': {'method': 'adaptive', 'gate': 3., 'scale': 3., 'trim': 4., 'mad_multiplier': 0.},
            'adaptive_trim': {'method': 'adaptive', 'gate': 3., 'scale': 3., 'trim': 4., 'mad_multiplier': 3.},
        }
        seeds, count = [20261031, 20261032], 30
    else:
        frozen = json.loads((out/'frozen.json').read_text())
        configs = frozen['configs']
        seeds, count = [20261121, 20261122, 20261123, 20261124], 60
    result = {'stage': args.stage, 'seeds': seeds, 'count_per_seed_scenario': count,
              'configs': configs, 'noise_levels_m': [1., 2.], 'runs': []}
    for noise in result['noise_levels_m']:
        for seed in seeds:
            data = dataset(seed, count, noise)
            metrics = {name: evaluate(data, cfg) for name, cfg in configs.items()}
            result['runs'].append({'seed': seed, 'noise_m': noise, 'metrics': metrics})
            print(json.dumps({'seed': seed, 'noise': noise,
                'cep90': {k: v['cep90_m'] for k, v in metrics.items()} }), flush=True)
    result['pooled'] = {}
    for noise in result['noise_levels_m']:
        grouped = [r for r in result['runs'] if r['noise_m'] == noise]
        pooled = {}
        for name in configs:
            pooled[name] = {s[0]: float(np.quantile([e for r in grouped for e in
                r['metrics'][name]['errors_m'][s[0]]], .9)) for s in SCENARIOS}
        result['pooled'][str(noise)] = pooled
    if args.stage == 'test':
        result['frozen_sha256'] = hashlib.sha256((out/'frozen.json').read_bytes()).hexdigest()
    (out/f'{args.stage}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['pooled'], indent=2))


if __name__ == '__main__':
    main()
