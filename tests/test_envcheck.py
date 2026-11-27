# encoding: utf-8
"""Tests of envcheck: the environment report and the batch checks (local, and Slurm with a stand-in)"""

import os

import pytest

import lvmdrp.cli.envcheck as envcheck
import lvmdrp.utils.cluster as cluster


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Required variables and directories in a temporary tree"""
    for var in ('LVMCORE_DIR', 'SAS_BASE_DIR', 'LVM_MASTER_DIR', 'LVM_SPECTRO_REDUX', 'LVM_DATA_S'):
        (tmp_path / var).mkdir()
        monkeypatch.setenv(var, str(tmp_path / var))
    (tmp_path / 'LVMCORE_DIR' / 'etc').mkdir()
    (tmp_path / 'LVMCORE_DIR' / 'etc' / 'exclude-exposures.yaml').write_text('excludes: []\n')
    monkeypatch.setattr(cluster, 'slurm_available', lambda: False)
    return tmp_path


def test_check_environment_no_problems(fake_env):
    assert envcheck.check_environment() == []


def test_check_environment_missing_and_unwritable(fake_env, monkeypatch):
    monkeypatch.delenv('LVM_MASTER_DIR')
    monkeypatch.setenv('LVM_DATA_S', str(fake_env / 'nowhere'))
    os.remove(fake_env / 'LVMCORE_DIR' / 'etc' / 'exclude-exposures.yaml')
    problems = envcheck.check_environment()
    assert 'missing: LVM_MASTER_DIR' in problems
    assert 'no raw data directory' in problems
    assert 'no exclude list in LVMCORE_DIR' in problems


def test_calibration_dir(fake_env, monkeypatch):
    from lvmdrp.core import constants
    monkeypatch.setattr(constants, 'configured_calib_dir', lambda config: None, raising=False)
    assert envcheck.calibration_dir() == (str(fake_env / 'LVM_MASTER_DIR'), 'LVM_MASTER_DIR')
    monkeypatch.setattr(constants, 'configured_calib_dir', lambda config: str(fake_env / 'calib_set'), raising=False)
    calib_dir, source = envcheck.calibration_dir()
    assert calib_dir == str(fake_env / 'calib_set') and source.startswith('calib_dir in lvmdrp.yaml')
    assert f'calibration directory {fake_env / "calib_set"} not found' in envcheck.check_environment()


def test_writable(tmp_path):
    assert envcheck._writable(str(tmp_path)) == 'exists, writable'
    assert envcheck._writable(str(tmp_path / 'a' / 'b')) == 'does not exist yet, can be created'


def test_envcheck_exit_codes(fake_env, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        envcheck.check_environment_variables([])
    assert stop.value.code == 0
    monkeypatch.delenv('SAS_BASE_DIR')
    with pytest.raises(SystemExit) as stop:
        envcheck.check_environment_variables([])
    assert stop.value.code == 1


def test_check_batch_local(tmp_path):
    assert envcheck.check_batch_local(str(tmp_path)) is True
    (log_dir,) = tmp_path.glob('envcheck-batch-*')
    assert sorted(line.split()[0] for line in open(log_dir / 'summary.txt')) == ['done', 'failed', 'skipped']


def test_check_batch_slurm_without_sbatch(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster, 'slurm_available', lambda: False)
    assert envcheck.check_batch_slurm(str(tmp_path)) is False
    assert list(tmp_path.iterdir()) == []                      # nothing written


def test_check_batch_slurm_with_stand_in(tmp_path, monkeypatch):
    """Slurm replaced by running the written array script's tasks with bash, then the final step"""
    import subprocess

    def fake_submit(jobs, log_dir, **kwargs):
        array_script, _ = cluster.write_slurm_scripts(jobs, log_dir=log_dir, label=kwargs['label'])
        for task in range(1, len(jobs) + 1):
            subprocess.run(['bash', array_script], env=dict(os.environ, SLURM_ARRAY_TASK_ID=str(task)))
        cluster.finish_slurm(log_dir)
        return True
    monkeypatch.setattr(cluster, 'slurm_available', lambda: True)
    monkeypatch.setattr(cluster, 'submit_slurm', fake_submit)
    assert envcheck.check_batch_slurm(str(tmp_path), timeout=1, poll=0.1) is True


def test_check_batch_slurm_submission_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster, 'slurm_available', lambda: True)
    monkeypatch.setattr(cluster, 'submit_slurm', lambda *a, **k: False)
    assert envcheck.check_batch_slurm(str(tmp_path)) is False
