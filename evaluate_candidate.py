"""Bounded development-set evaluation tool for the research agent."""
import argparse
import json
from pathlib import Path
from experiment import cases, evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--loss', required=True, choices=['linear', 'huber', 'soft_l1', 'cauchy'])
    p.add_argument('--scale', required=True, type=float)
    p.add_argument('--reason', required=True)
    p.add_argument('--output', default='results/live_development.jsonl')
    a = p.parse_args()
    if not 0.5 <= a.scale <= 20:
        p.error('scale must be in [0.5, 20] metres')
    config = {'loss': a.loss, 'scale': a.scale}
    metrics = evaluate(cases(20261010, 30), config)
    record = {'mode': 'external_research_agent_tool', 'config': config,
              'reason': a.reason, 'development_metrics': metrics}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'reason'}, ensure_ascii=False))
