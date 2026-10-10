"""Small mathematical checks, rather than tests that simply mirror the code."""
import json
from pathlib import Path
import numpy as np
from experiment import locate

truth = np.array([2., -3.])
anchors = np.array([[-20., -20.], [20., -20.], [20., 20.], [-20., 20.]])
case = {'anchors': anchors, 'distances': np.linalg.norm(anchors-truth, axis=1)}
for loss in ['linear', 'huber', 'soft_l1', 'cauchy']:
    prediction, _ = locate(case, {'loss': loss, 'scale': 3.})
    assert np.linalg.norm(prediction-truth) < 1e-5, (loss, prediction)
summary = json.loads(Path('results/summary.json').read_text())
assert summary['development_seed'] != summary['test_seed']
records = [json.loads(s) for s in Path('results/iterations.jsonl').read_text().splitlines()]
assert len(records) == summary['rounds']
best = min(records, key=lambda r: r['selection_score'])
assert best['config'] == summary['selected']
for method in summary['test'].values():
    for scenario, errors in method['errors_m'].items():
        assert len(errors) == summary['cases_per_scenario_per_split']
        assert np.isfinite(errors).all()
        assert abs(np.quantile(errors, 0.9)-method['by_scenario'][scenario]['cep90_m']) < 1e-10
print('PASS: noiseless recovery, distinct split seeds, selection audit, CEP90 recomputation')
