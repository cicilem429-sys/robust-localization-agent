"""Check mathematical recovery, truth isolation, split provenance and metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_research import adaptive_locate

p = Path('results/adaptive')
frozen = json.loads((p/'frozen.json').read_text())
config = frozen['configs'][frozen['selected']]
truth = np.array([2., -3.])
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
anchors = truth + 30 * np.column_stack([np.cos(angles), np.sin(angles)])
observations = {'anchors': anchors, 'distances': np.full(12, 30.)}
prediction, diagnostic = adaptive_locate(observations, config)
assert diagnostic['route'] == 'least_squares'
assert np.linalg.norm(prediction-truth) < 1e-5
# A hidden truth value must not influence the estimator.
other, _ = adaptive_locate({**observations, 'truth': np.array([1e9, -1e9])}, config)
assert np.array_equal(prediction, other)
contaminated = {**observations, 'distances': observations['distances'].copy()}
contaminated['distances'][[0, 4, 8]] += 20
prediction, diagnostic = adaptive_locate(contaminated, config)
assert diagnostic['route'] == 'trim_refit'
assert diagnostic['removed'] == 3
assert np.linalg.norm(prediction-truth) < 1e-5

dev = json.loads((p/'development.json').read_text())
test = json.loads((p/'test.json').read_text())
assert set(dev['seeds']).isdisjoint(test['seeds'])
assert test['seeds'] == frozen['test_seeds']
assert test['configs'] == frozen['configs']
assert test['frozen_sha256'] == hashlib.sha256((p/'frozen.json').read_bytes()).hexdigest()
for result in [dev, test]:
    for run in result['runs']:
        for metrics in run['metrics'].values():
            for scenario, errors in metrics['errors_m'].items():
                assert len(errors) == result['count_per_seed_scenario']
                assert np.isfinite(errors).all()
                assert np.isclose(np.quantile(errors, .9), metrics['cep90_m'][scenario])
    for noise, methods in result['pooled'].items():
        for name, scenarios in methods.items():
            for scenario, cep in scenarios.items():
                errors = [e for r in result['runs'] if r['noise_m'] == float(noise)
                          for e in r['metrics'][name]['errors_m'][scenario]]
                assert np.isclose(np.quantile(errors, .9), cep)

# Stratified paired bootstrap: resample cases within each independent seed.
# Approximate empirical intervals conditional on four seeds, not a universal guarantee.
rng = np.random.default_rng(20261201)
intervals = {}
for noise in test['noise_levels_m']:
    runs = [r for r in test['runs'] if r['noise_m'] == noise]
    baseline = np.array([r['metrics']['least_squares']['errors_m']['clean'] for r in runs])
    selected = np.array([r['metrics'][frozen['selected']]['errors_m']['clean'] for r in runs])
    ratios = []
    for _ in range(2000):
        indices = rng.integers(0, baseline.shape[1], baseline.shape)
        a = np.take_along_axis(baseline, indices, axis=1).ravel()
        b = np.take_along_axis(selected, indices, axis=1).ravel()
        ratios.append(np.quantile(b, .9)/np.quantile(a, .9))
    intervals[str(noise)] = {'clean_cep90_ratio_95_percentile_interval':
                            np.quantile(ratios, [.025, .975]).tolist(),
                            'replicates': 2000, 'bootstrap_seed': 20261201}
(p/'verification.json').write_text(json.dumps({'checks': 'passed', 'intervals': intervals}, indent=2))
print('PASS: noiseless recovery, exact gross-error rejection, truth isolation, frozen splits, CEP90 recomputation')
print(json.dumps(intervals, indent=2))
