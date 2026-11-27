# encoding: utf-8
"""Tests of the batch runs (Slurm or local) of lvmdrp.utils.cluster and the outcomes of drp run"""

import os

import pandas as pd
import pytest
from click.testing import CliRunner

import lvmdrp.main as main
import lvmdrp.utils.cluster as cluster
from lvmdrp.main import run_outcomes, RUN_EXIT_CODES


# --- job lists -----------------------------------------------------------------------------------

def test_science_jobs_per_exposure_adds_skip_drpall():
    jobs = cluster.science_jobs(expnums=[19475, 19476], drp_options='--from-frame')
    assert [name for name, _ in jobs] == ['19475', '19476']
    assert jobs[0][1] == 'umask 002 && drp run -e 19475 --from-frame --skip-drpall'


def test_science_jobs_keeps_given_skip_drpall():
    jobs = cluster.science_jobs(expnums='19475-19477', drp_options='-da')
    assert len(jobs) == 3
    assert jobs[2][1].endswith('drp run -e 19477 -da')


def test_science_jobs_from_file(tmp_path):
    expfile = tmp_path / 'exps.txt'
    expfile.write_text('19475\n19476\n')
    jobs = cluster.science_jobs(expnums=str(expfile))
    assert [name for name, _ in jobs] == ['19475', '19476']


def test_science_jobs_open_range_gives_nothing():
    assert cluster.science_jobs(expnums='19475-') == []


def test_science_jobs_per_night():
    jobs = cluster.science_jobs(mjds=60478)
    assert jobs == [('60478', 'umask 002 && drp run -m 60478 --skip-drpall')]


def test_calibration_jobs_from_epochs_only_where_accepted():
    jobs = cluster.calibration_jobs([60465], routine='long-term', from_epochs=True)
    assert jobs == [('long-term-60465', 'umask 002 && drp calibrations long-term --from-epochs -m 60465')]
    jobs = cluster.calibration_jobs([60465], routine='fiberflats', from_epochs=True)
    assert jobs == [('fiberflats-60465', 'umask 002 && drp calibrations fiberflats -m 60465')]


# --- outcomes ------------------------------------------------------------------------------------

def test_run_outcomes_codes():
    assert run_outcomes({1: 'done', 2: 'done'})[1] == RUN_EXIT_CODES['done']
    assert run_outcomes({1: 'done', 2: 'skipped'})[1] == RUN_EXIT_CODES['skipped']
    assert run_outcomes({1: 'skipped', 2: 'failed'})[1] == RUN_EXIT_CODES['failed']
    assert run_outcomes({})[1] == RUN_EXIT_CODES['done']


def test_run_outcomes_missing_exposure_fails():
    groups, code = run_outcomes({1: 'done'}, requested=[1, 2])
    assert groups == {'done': [1], 'skipped': [], 'failed': [2]}
    assert code == RUN_EXIT_CODES['failed']


def test_outcome_of_exit_codes():
    assert cluster.outcome(0) == 'done'
    assert cluster.outcome(3) == 'skipped'
    assert cluster.outcome(1) == 'failed'
    assert cluster.outcome(-9) == 'failed'


# --- run_drp outcomes ----------------------------------------------------------------------------

@pytest.fixture
def fake_night(tmp_path, monkeypatch):
    """A night with science exposures 1 (GOOD), 2 (BAD), 3 (GOOD, excluded), 4 (GOOD)"""
    mjd = 60478
    (tmp_path / str(mjd)).mkdir()
    monkeypatch.setenv('LVM_DATA_S', str(tmp_path))
    frames = pd.DataFrame({'expnum': [1, 2, 3, 4], 'tileid': 11111, 'mjd': mjd, 'imagetyp': 'object',
                           'qaqual': ['GOOD', 'BAD', 'GOOD', 'GOOD'], 'camera': 'b1'})
    monkeypatch.setattr(main, 'update_metadata', lambda mjd: None)
    monkeypatch.setattr(main, 'get_frames_metadata', lambda mjd: frames.copy())
    monkeypatch.setattr(main, 'load_excludes', lambda: None)
    monkeypatch.setattr(main, 'remove_excluded_frames', lambda df, ex: df[df['expnum'] != 3])
    monkeypatch.setattr(main, 'create_status_file', lambda *a, **k: None)
    monkeypatch.setattr(main, 'status_file_exists', lambda *a, **k: False)
    monkeypatch.setattr(main, 'start_logging', lambda *a, **k: 'log')
    monkeypatch.setattr(main, 'update_error_file', lambda *a, **k: None)
    monkeypatch.setattr(main, 'get_config_options', lambda *a, **k: {})
    return mjd


def test_run_drp_outcomes(fake_night, monkeypatch):
    def fake_reduction(expnum, **kwargs):
        if expnum == 4:
            return None          # stopped early
        return True
    monkeypatch.setattr(main, 'science_reduction', fake_reduction)
    outcomes = main.run_drp(mjd=fake_night)
    assert outcomes == {1: 'done', 2: 'skipped', 3: 'skipped', 4: 'failed'}


def test_run_drp_exception_is_failed(fake_night, monkeypatch):
    def fake_reduction(expnum, **kwargs):
        raise RuntimeError('boom')
    monkeypatch.setattr(main, 'science_reduction', fake_reduction)
    outcomes = main.run_drp(mjd=fake_night, expnum=1)
    assert outcomes == {1: 'failed'}


def test_drp_run_exit_code(fake_night, monkeypatch):
    from lvmdrp.cli.drp import cli
    monkeypatch.setattr('lvmdrp.cli.drp.run_drp', lambda **kwargs: {1: 'done', 2: 'skipped'})
    result = CliRunner().invoke(cli, ['run', '-m', str(fake_night)])
    assert result.exit_code == RUN_EXIT_CODES['skipped']
    monkeypatch.setattr('lvmdrp.cli.drp.run_drp', lambda **kwargs: {1: 'done'})
    result = CliRunner().invoke(cli, ['run', '-el', '1', '-el', '5'])
    assert result.exit_code == RUN_EXIT_CODES['failed']       # 5 not found


# --- local runs ----------------------------------------------------------------------------------

def test_run_local(tmp_path):
    jobs = [('a', 'echo hello; exit 0'), ('b', 'exit 3'), ('c', 'echo $OMP_NUM_THREADS; exit 1')]
    results = cluster.run_local(jobs, np=2, log_dir=str(tmp_path / 'logs'), stagger=0)
    assert [r['outcome'] for r in results] == ['done', 'skipped', 'failed']
    assert open(tmp_path / 'logs' / 'a.log').read().splitlines()[-1] == 'hello'
    if 'OMP_NUM_THREADS' not in os.environ:
        assert open(tmp_path / 'logs' / 'c.log').read().splitlines()[-1] == '1'
    summary = open(tmp_path / 'logs' / 'summary.txt').read().splitlines()
    assert sorted(line.split()[0] for line in summary) == ['a', 'b', 'c']
    assert cluster.report(results) == RUN_EXIT_CODES['failed']


def test_run_cluster_local(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(cluster, 'nights_of', lambda expnums: [60478])
    monkeypatch.setattr(cluster, 'regenerate_metadata', lambda mjds: calls.setdefault('metadata', mjds))
    monkeypatch.setattr(cluster, 'run_local', lambda jobs, **kw: calls.setdefault('jobs', jobs)
                        and [{'name': n, 'outcome': 'done', 'exit': 0} for n, _ in jobs])
    monkeypatch.setattr(main, 'create_drpall', lambda: calls.setdefault('drpall', True))
    code = cluster.run_cluster(expnums=[19475, 19476], local=True, drp_options='--from-frame')
    assert code == 0
    assert calls['metadata'] == [60478]
    assert [n for n, _ in calls['jobs']] == ['19475', '19476']
    assert calls['drpall'] is True


def test_run_cluster_slurm_unavailable(monkeypatch):
    monkeypatch.setattr(cluster, 'slurm_available', lambda: False)
    monkeypatch.setattr(cluster, 'nights_of', lambda expnums: [])
    monkeypatch.setattr(cluster, 'regenerate_metadata', lambda mjds: pytest.fail('metadata regenerated'))
    assert cluster.run_cluster(expnums=[19475]) == 1


# --- Slurm runs (sbatch job array) ---------------------------------------------------------------

def test_write_slurm_scripts(tmp_path):
    jobs = [('19475', 'umask 002 && drp run -e 19475 --from-frame --skip-drpall'), ('19476', 'exit 3')]
    array_script, finish_script = cluster.write_slurm_scripts(jobs, log_dir=str(tmp_path), alloc='sdss-np',
                                                              walltime='01:00:00', mem='16G', max_running=10,
                                                              drpall=True)
    text = open(array_script).read()
    for line in ('#SBATCH --account=sdss-np', '#SBATCH --partition=sdss-np', '#SBATCH --qos=sdss-np',
                 '#SBATCH --time=01:00:00', '#SBATCH --mem=16G', '#SBATCH --ntasks=1', '#SBATCH --array=1-2%10'):
        assert line in text.splitlines()
    assert open(tmp_path / 'jobs.txt').read().splitlines()[0] == f'19475\t{jobs[0][1]}'
    finish = open(finish_script).read()
    assert f"python -m lvmdrp.utils.cluster finish '{tmp_path}' --drpall" in finish


def _run_tasks(array_script, ntasks, env_extra=None):
    """Runs the tasks of an array script with bash, as Slurm would (the #SBATCH lines are comments)"""
    import subprocess
    codes = []
    for task in range(1, ntasks + 1):
        env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(task), **(env_extra or {}))
        codes.append(subprocess.run(['bash', array_script], env=env).returncode)
    return codes


def test_slurm_tasks_and_finish(tmp_path, monkeypatch):
    jobs = [('a', 'echo "hello world"; exit 0'), ('b', 'exit 3'), ('c', 'echo $OMP_NUM_THREADS; exit 1'),
            ('d', 'exit 0')]
    array_script, _ = cluster.write_slurm_scripts(jobs, log_dir=str(tmp_path))
    codes = _run_tasks(array_script, 3)                       # task 4 never runs (killed)
    assert codes == [0, 3, 1]
    assert open(tmp_path / 'a.log').read().splitlines() == ['# echo "hello world"; exit 0', 'hello world']
    if 'OMP_NUM_THREADS' not in os.environ:
        assert open(tmp_path / 'c.log').read().splitlines()[-1] == '1'
    assert open(tmp_path / 'status' / 'b').read().split()[:4] == ['b', 'exit', '3', 'skipped']

    drpall = []
    monkeypatch.setattr(main, 'create_drpall', lambda: drpall.append(True))
    code = cluster.finish_slurm(str(tmp_path), drpall=True)
    assert code == RUN_EXIT_CODES['failed']
    summary = open(tmp_path / 'summary.txt').read().splitlines()
    assert [line.split()[0] for line in summary] == ['a', 'b', 'c', 'd']
    assert [line.split()[3] for line in summary[:3]] == ['done', 'skipped', 'failed']
    assert summary[3].startswith('d exit none failed') and 'slurm-*_4.out' in summary[3]
    assert drpall == [True]


def test_finish_slurm_main_all_done(tmp_path, monkeypatch):
    array_script, _ = cluster.write_slurm_scripts([('a', 'exit 0'), ('b', 'exit 0')], log_dir=str(tmp_path))
    _run_tasks(array_script, 2)
    monkeypatch.setattr(main, 'create_drpall', lambda: pytest.fail('drpall made without --drpall'))
    assert cluster.main(['finish', str(tmp_path)]) == 0


def test_finish_slurm_main_exits_0_when_jobs_failed(tmp_path):
    """The final job succeeds when it has done its work, even if some jobs failed"""
    array_script, _ = cluster.write_slurm_scripts([('a', 'exit 0'), ('b', 'exit 1')], log_dir=str(tmp_path))
    _run_tasks(array_script, 2)
    assert cluster.main(['finish', str(tmp_path)]) == 0
    assert [line.split()[3] for line in open(tmp_path / 'summary.txt')] == ['done', 'failed']


class _FakeSbatch:
    """Stands in for subprocess.run of sbatch: returns job ids, or fails on the n-th call"""
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, args, **kwargs):
        import subprocess
        self.calls.append(args)
        if self.fail_on == len(self.calls):
            return subprocess.CompletedProcess(args, 1, stdout='', stderr='sbatch: error: Invalid account')
        return subprocess.CompletedProcess(args, 0, stdout=f'{1000 + len(self.calls)};notchpeak\n', stderr='')


def test_submit_slurm_array_then_finish(tmp_path, monkeypatch):
    fake = _FakeSbatch()
    monkeypatch.setattr(cluster, 'slurm_available', lambda: True)
    monkeypatch.setattr(cluster.subprocess, 'run', fake)
    assert cluster.submit_slurm([('a', 'exit 0')], log_dir=str(tmp_path), drpall=True) is True
    assert fake.calls[0][:2] == ['sbatch', '--parsable'] and fake.calls[0][2].endswith('lvm_cluster_run.sbatch')
    assert fake.calls[1][2] == '--dependency=afterany:1001'
    assert fake.calls[1][3].endswith('lvm_cluster_run-finish.sbatch')


def test_submit_slurm_not_submitted(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster, 'slurm_available', lambda: True)
    monkeypatch.setattr(cluster.subprocess, 'run', lambda *a, **k: pytest.fail('sbatch called'))
    assert cluster.submit_slurm([('a', 'exit 0')], log_dir=str(tmp_path), submit=False) is True
    assert (tmp_path / 'lvm_cluster_run.sbatch').is_file()


def test_run_cluster_sbatch_failure_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster, 'slurm_available', lambda: True)
    monkeypatch.setattr(cluster, 'nights_of', lambda expnums: [60478])
    monkeypatch.setattr(cluster, 'regenerate_metadata', lambda mjds: None)
    monkeypatch.setattr(cluster.subprocess, 'run', _FakeSbatch(fail_on=1))
    assert cluster.run_cluster(expnums=[19475], log_dir=str(tmp_path)) == 1
    fake = _FakeSbatch(fail_on=2)                              # array submitted, final job not
    monkeypatch.setattr(cluster.subprocess, 'run', fake)
    assert cluster.run_cluster(expnums=[19475], log_dir=str(tmp_path)) == 1
    assert len(fake.calls) == 2


def test_run_cluster_slurm_options(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cluster, 'slurm_available', lambda: True)
    monkeypatch.setattr(cluster, 'nights_of', lambda expnums: [])
    monkeypatch.setattr(cluster, 'submit_slurm', lambda jobs, **kw: seen.update(kw) or True)
    assert cluster.run_cluster(expnums=[19475], nodes=1, ppn=4, walltime='01:00:00', mem='8G',
                               log_dir=str(tmp_path)) == 0
    assert seen['max_running'] == 4 and seen['walltime'] == '01:00:00' and seen['mem'] == '8G'
    assert seen['drpall'] is True and seen['alloc'] == 'sdss-np'


def test_run_cluster_dry_run_runs_nothing(monkeypatch):
    monkeypatch.setattr(cluster, 'nights_of', lambda expnums: [60478])
    monkeypatch.setattr(cluster, 'regenerate_metadata', lambda mjds: pytest.fail('metadata regenerated'))
    monkeypatch.setattr(cluster, 'run_local', lambda *a, **k: pytest.fail('jobs run'))
    assert cluster.run_cluster(expnums=[19475], local=True, dry_run=True) == 0


def test_cluster_cli_local(monkeypatch):
    from lvmdrp.cli.drp import cli
    seen = {}

    def fake_run_cluster(**kwargs):
        seen.update(kwargs)
        return 3
    monkeypatch.setattr('lvmdrp.cli.drp.run_cluster', fake_run_cluster)
    result = CliRunner().invoke(cli, ['cluster', '-el', '19475', '-el', '19476', '--local', '--np', '4',
                                      '--drp-options', '--from-frame'])
    assert result.exit_code == 3
    assert seen['local'] is True and seen['np'] == 4 and seen['expnums'] == (19475, 19476)
    result = CliRunner().invoke(cli, ['cluster', '-e', '19475', '--mem', '8G'])
    assert seen['mem'] == '8G' and seen['local'] is False


def test_run_drp_exposure_list_visits_each_night_once(fake_night, monkeypatch):
    nights = {1: 60478, 4: 60478}

    def fake_mjd(exp):
        if exp not in nights:
            raise ValueError(f'no raw frame found for exposure number {exp}')
        return [nights[exp]]
    monkeypatch.setattr(main, 'mjd_from_expnum', fake_mjd)
    reduced = []
    monkeypatch.setattr(main, 'science_reduction', lambda expnum, **kw: reduced.append(expnum) or True)
    outcomes = main.run_drp(mjd=None, expnum=[1, 4, 99])
    assert sorted(reduced) == [1, 4]                      # each once, although two exposures share the night
    assert outcomes == {1: 'done', 4: 'done'}
    groups, code = run_outcomes(outcomes, requested=[1, 4, 99])
    assert groups['failed'] == [99] and code == RUN_EXIT_CODES['failed']


def test_ancillary_dir_of(monkeypatch, tmp_path):
    """ --clean-ancillary finds the ancillary directory from the metadata, also when -1d/--from-frame skip extraction """
    monkeypatch.setenv('LVM_SPECTRO_REDUX', str(tmp_path))
    meta = pd.DataFrame([{'tileid': 1054275, 'mjd': 60478, 'expnum': 19475, 'camera': cam, 'imagetyp': 'object'}
                         for cam in ('b1', 'r1')])
    out = main.ancillary_dir_of(meta, drpver='test')
    assert out.startswith(str(tmp_path / 'test'))
    assert out.endswith(os.path.join('1054275', '60478', 'ancillary'))
