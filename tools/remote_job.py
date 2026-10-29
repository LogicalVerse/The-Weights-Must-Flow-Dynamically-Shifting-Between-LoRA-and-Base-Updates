#!/usr/bin/env python3
"""Detached, single-job supervisor for Linux SSH workers. Standard library only."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def lock_held(root):
    """File lock is authoritative; no external PID-based signaling/reuse risk."""
    with (Path(root) / 'running.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def terminate_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


def supervise(job_dir):
    job_dir = Path(job_dir)
    with (job_dir / 'running.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 79
        request = read_json(job_dir / 'request.json')
        state = dict(status='running', pid=os.getpid(), run_token=uuid.uuid4().hex,
                     started_at=now(), request_digest=request['request_digest'])
        write_json(job_dir / 'status.json', state)
        stopped = [False]
        def stop_handler(*_):
            stopped[0] = True
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        child = None
        try:
            env = os.environ.copy()
            additions = json.loads(Path(request['env_file']).read_text())
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in additions.items()):
                raise ValueError('Environment values must be strings')
            env.update(additions)
            deadline = time.monotonic() + request['timeout_hours'] * 3600
            with (job_dir / 'output.log').open('a', buffering=1) as log:
                child = subprocess.Popen(request['command'], cwd=request['cwd'], env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                state['child_pid'] = child.pid
                while child.poll() is None:
                    state['heartbeat_at'] = now()
                    write_json(job_dir / 'status.json', state)
                    stop_request = read_json(job_dir / 'stop_request.json')
                    if stop_request.get('run_token') == state['run_token']:
                        stopped[0] = True
                    if stopped[0] or time.monotonic() >= deadline:
                        state['status'] = 'stopped' if stopped[0] else 'timed_out'
                        terminate_child(child)
                        break
                    time.sleep(1)
                state['returncode'] = child.wait()
            if state['status'] == 'running':
                state['status'] = 'completed' if state['returncode'] == 0 else 'failed'
        except Exception as exc:
            if child is not None:
                terminate_child(child)
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        state['finished_at'] = now()
        write_json(job_dir / 'status.json', state)
        return 0 if state['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'stop', '_run'])
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--cwd')
    parser.add_argument('--env-file')
    parser.add_argument('--timeout-hours', type=float, default=16)
    parser.add_argument('--retry', action='store_true')
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    command = argv[split + 1:] if split < len(argv) else []
    args = parser.parse_args(argv[:split])
    root = Path(args.job_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == '_run':
        return supervise(root)
    state = read_json(root / 'status.json')
    if args.action == 'status':
        if state.get('status') == 'running' and not lock_held(root):
            state = read_json(root / 'status.json')
            if state.get('status') == 'running':
                state['status'] = 'interrupted'
        print(json.dumps(state or {'status': 'not_started'}))
        return 0
    if args.action == 'stop':
        if state.get('status') == 'running' and lock_held(root):
            write_json(root / 'stop_request.json', {'run_token': state['run_token'], 'requested_at': now()})
            print(json.dumps({'status': 'stop_requested'}))
        else:
            print(json.dumps({'status': 'not_running'}))
        return 0
    if not command or not args.cwd or not args.env_file or args.timeout_hours <= 0:
        parser.error('start requires --cwd, --env-file, positive timeout and -- COMMAND')
    request = dict(command=command, cwd=str(Path(args.cwd).resolve()),
                   env_file=str(Path(args.env_file).resolve()), timeout_hours=args.timeout_hours)
    request['request_digest'] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    with (root / 'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX)
        old_request = read_json(root / 'request.json')
        state = read_json(root / 'status.json')
        if old_request and old_request.get('request_digest') != request['request_digest']:
            raise RuntimeError('Job specification changed: use a new job directory')
        if state.get('status') == 'running' and lock_held(root):
            print(json.dumps(state)); return 0
        if state.get('status') == 'completed':
            print(json.dumps(state)); return 0
        if state and not args.retry:
            raise RuntimeError('Previous attempt did not complete. Inspect logs; use --retry explicitly.')
        write_json(root / 'request.json', request)
        if state:
            write_json(root / f"previous_status_{time.time_ns()}.json", state)
            (root / 'status.json').unlink(missing_ok=True)
        with (root / 'supervisor.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_run',
                                     '--job-dir', str(root)], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(30):
            state = read_json(root / 'status.json')
            if state:
                print(json.dumps(state)); return 0
            if proc.poll() is not None:
                raise RuntimeError('Supervisor failed to start; inspect supervisor.log')
            time.sleep(.1)
        raise RuntimeError('No startup acknowledgement. Check status before retrying.')


if __name__ == '__main__':
    raise SystemExit(main())
