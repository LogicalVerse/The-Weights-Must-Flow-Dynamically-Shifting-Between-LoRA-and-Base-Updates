"""Local Linux tests for the launcher's non-GPU behavior. No network needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

KIT = Path(__file__).resolve().parents[1]
RUNNER = KIT / 'tools/remote_job.py'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = self.root / 'env.json'
        self.env.write_text(json.dumps({'AUTOMODE_TEST_VALUE': 'passed'}))
        self.job = self.root / 'job'

    def tearDown(self):
        self.call('stop', check=False)
        time.sleep(.05)
        self.temp.cleanup()

    def call(self, action, more=(), check=True):
        p = subprocess.run([sys.executable, str(RUNNER), action, '--job-dir', str(self.job), *more],
                           capture_output=True, text=True)
        if check and p.returncode:
            self.fail(p.stderr)
        return p

    def start(self, source, timeout=.02, check=True):
        return self.call('start', ['--cwd', str(self.root), '--env-file', str(self.env),
                                  '--timeout-hours', str(timeout), '--', sys.executable, '-c', source], check)

    def wait_terminal(self):
        for _ in range(100):
            state = json.loads(self.call('status').stdout)
            if state['status'] != 'running':
                return state
            time.sleep(.1)
        self.fail('Job did not finish')

    def test_success_environment_and_skip_completed(self):
        source = "import os,pathlib; p=pathlib.Path('result.txt'); p.write_text(p.read_text()+'x' if p.exists() else os.environ['AUTOMODE_TEST_VALUE'])"
        self.start(source)
        self.assertEqual(self.wait_terminal()['status'], 'completed')
        self.start(source)
        self.assertEqual((self.root / 'result.txt').read_text(), 'passed')

    def test_failure_is_not_silently_retried(self):
        self.start('raise RuntimeError("intentional test failure")')
        self.assertEqual(self.wait_terminal()['status'], 'failed')
        p = self.start('raise RuntimeError("intentional test failure")', check=False)
        self.assertNotEqual(p.returncode, 0)

    def test_stop(self):
        self.start('import time; time.sleep(60)')
        self.call('stop')
        self.assertEqual(self.wait_terminal()['status'], 'stopped')

    def test_duplicate_running_launch_is_idempotent(self):
        source = 'import time; time.sleep(60)'
        first = json.loads(self.start(source).stdout)
        second = json.loads(self.start(source).stdout)
        self.assertEqual(first['run_token'], second['run_token'])
        self.call('stop')
        self.assertEqual(self.wait_terminal()['status'], 'stopped')

    def test_timeout(self):
        self.start('import time; time.sleep(60)', timeout=.00015)
        self.assertEqual(self.wait_terminal()['status'], 'timed_out')

    def test_changed_request_refused(self):
        self.start('print("first")')
        self.wait_terminal()
        self.assertNotEqual(self.start('print("second")', check=False).returncode, 0)

    def test_notebook_and_manifest(self):
        nb = json.loads((KIT / 'notebooks/AutoMode_Four_GPU_Colab.ipynb').read_text())
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(cell['source'], f'cell_{i}', 'exec')
                self.assertEqual(cell['outputs'], [])
        manifest = json.loads((KIT / 'configs/rebuttal_core.json').read_text())
        jobs = manifest['jobs']
        self.assertEqual(len({j['job_id'] for j in jobs}), 15)
        self.assertEqual([sum(j['worker_id']==i for j in jobs) for i in range(4)], [4,4,4,3])
        for method in ['automode', 'select_once', 'fixed_mixed', 'static_qv', 'lora']:
            self.assertEqual(sorted(j['overrides']['seed'] for j in jobs if j['overrides']['method']==method), [8,25,42])
        embedded = next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'REMOTE_JOB_SOURCE =' in c['source'])
        namespace = {}; exec(embedded, namespace)
        self.assertEqual(namespace['CORE_MANIFEST'], manifest)
        self.assertEqual(namespace['REMOTE_JOB_SOURCE'], RUNNER.read_text())


if __name__ == '__main__':
    unittest.main()
