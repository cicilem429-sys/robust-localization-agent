"""Start teacher's Stage 2 controller with this project's tool contract.

This is an integration launcher, not a copy or modification of upstream code.
Requires configured Codex CLI and the upstream checkout. Never uses shell=True.
"""
import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--teacher-repo', required=True, type=Path)
    p.add_argument('--run-id', default='crowd-localization-001')
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    project = Path(__file__).resolve().parent
    controller = a.teacher_repo.resolve()/'mathematical-concept-agent'/'agent.py'
    if not controller.is_file():
        p.error('Upstream mathematical-concept-agent/agent.py not found')
    prompt = (project/'docs'/'research_prompt.md').read_text()
    prompt += '\n本项目绝对路径：'+str(project)+'\n工具请使用当前 Python：'+sys.executable
    command = [sys.executable, str(controller), 'run', a.run_id, '--prompt', prompt]
    if a.once:
        command.append('--once')
    subprocess.run(command, cwd=controller.parent, check=True)
