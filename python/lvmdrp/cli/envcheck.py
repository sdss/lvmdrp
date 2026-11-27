#!/usr/bin/env python
# encoding: utf-8
#
"""Checks the environment of the LVM DRP, before reducing anything

``envcheck`` reports which lvmdrp is used, where the DRP reads its inputs and writes its
outputs, and whether batch runs can use Slurm here. It changes nothing and works on any machine.

``envcheck --batch`` also runs three dummy jobs (exit 0, 3 and 1) through the local batch code of
``drp cluster --local``, and ``envcheck --slurm`` runs them through Slurm with ``sbatch``, as
``drp cluster`` does, and waits for the result. Neither runs the DRP itself.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from lvmdrp import log

REQUIRED_VARS = ["LVMCORE_DIR", "SAS_BASE_DIR", "LVM_MASTER_DIR"]

#: the dummy jobs of the batch checks: the job name is the outcome it should have
CHECK_JOBS = [('done', 'echo "running on $(hostname)"; exit 0'), ('skipped', 'exit 3'), ('failed', 'exit 1')]


def _git(directory: str, *args) -> str:
    """Returns the output of a git command in `directory`, or '' if it is not a git checkout"""
    try:
        return subprocess.run(['git', '-C', directory] + list(args), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def _writable(directory: str) -> str:
    """Describes whether `directory` exists and can be written, or else could be created"""
    if os.path.isdir(directory):
        return 'exists, writable' if os.access(directory, os.W_OK) else 'exists, NOT writable'
    parent = os.path.dirname(os.path.abspath(directory))
    while parent and not os.path.isdir(parent):
        parent = os.path.dirname(parent)
    if parent and os.access(parent, os.W_OK):
        return 'does not exist yet, can be created'
    return f'does not exist, and {parent} is NOT writable'


def calibration_dir() -> tuple:
    """Returns the calibration directory ``drp`` uses without --calib-dir, and where it comes from

    Where lvmdrp names a calibration directory in lvmdrp.yaml (``calib_dir``, sdss/lvmdrp#272), it
    replaces $LVM_MASTER_DIR; otherwise $LVM_MASTER_DIR is used.

    Returns
    -------
    tuple[str, str]
        the directory (None if neither is set) and its source
    """
    import lvmdrp
    from lvmdrp.core import constants
    configured = getattr(constants, 'configured_calib_dir', None)
    calib_dir = configured(getattr(lvmdrp, 'config', None)) if configured else None
    if calib_dir:
        return calib_dir, 'calib_dir in lvmdrp.yaml, replacing LVM_MASTER_DIR; drp run --calib-dir overrides it'
    return os.getenv('LVM_MASTER_DIR'), 'LVM_MASTER_DIR'


def check_environment() -> list:
    """Reports the environment of the DRP and returns the problems found

    Returns
    -------
    list[str]
        the problems; a missing required variable starts with 'missing:'
    """
    import lvmdrp
    from sdsstools import get_package_version
    from lvmdrp.utils.cluster import slurm_available

    problems = []

    def report(label, value, problem=None):
        if problem:
            problems.append(problem)
            log.warning(f'{label:24s} {value}   <-- {problem}')
        else:
            log.info(f'{label:24s} {value}')

    # the code
    package_dir = os.path.dirname(lvmdrp.__file__)
    log.info('lvmdrp')
    report('  imported from', package_dir)
    branch = _git(package_dir, 'rev-parse', '--abbrev-ref', 'HEAD')
    if branch:
        dirty = ' (with uncommitted changes)' if _git(package_dir, 'status', '--porcelain', '--untracked-files=no') else ''
        report('  git checkout', f'branch {branch}, commit {_git(package_dir, "rev-parse", "--short=8", "HEAD")}{dirty}')
    else:
        report('  git checkout', 'none (installed package)')
    report('  python', f'{sys.version.split()[0]} in {sys.prefix}')

    # required variables
    log.info('required environment variables')
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if value is None:
            report(f'  {var}', 'NOT SET', f'missing: {var}')
        elif not os.path.isdir(value):
            report(f'  {var}', value, f'{var} is not a directory')
        else:
            report(f'  {var}', value)

    # outputs
    log.info('outputs')
    label = lvmdrp.__version__
    code_version = get_package_version(path=lvmdrp.__file__, package_name='lvmdrp')
    drpver = (getattr(lvmdrp, 'config', None) or {}).get('drpver')
    if os.getenv('LVMDRP_VERSION'):
        source = 'from LVMDRP_VERSION'
    elif drpver and label == str(drpver):
        source = 'from drpver in lvmdrp.yaml; LVMDRP_VERSION not set'
    else:
        source = 'the code version; LVMDRP_VERSION not set'
    report('  version label', f'{label}  ({source}; names the redux directory)')
    report('  code version', f'{code_version}  (setup.cfg; the label only when nothing else sets it)')
    redux = os.getenv('LVM_SPECTRO_REDUX')
    if redux:
        out_dir = os.path.join(redux, label)
        state = _writable(out_dir)
        report('  reductions go to', f'{out_dir}  ({state})', None if 'NOT' not in state else f'{out_dir}: {state}')
    else:
        report('  reductions go to', 'unknown', 'LVM_SPECTRO_REDUX is not set (it comes from SAS_BASE_DIR)')

    # inputs
    log.info('inputs')
    raw = os.getenv('LVM_DATA_S')
    if raw and os.path.isdir(raw):
        report('  raw data (LVM_DATA_S)', raw)
    else:
        report('  raw data (LVM_DATA_S)', raw or 'NOT SET', 'no raw data directory')
    calib_dir, calib_source = calibration_dir()
    if calib_dir:
        problem = None if os.path.isdir(calib_dir) else f'calibration directory {calib_dir} not found'
        report('  calibrations', f'{calib_dir}  ({calib_source})', problem)
        cache = os.path.join(calib_dir, 'gaia_cache')
        report('  Gaia XP cache', f'{cache}  ({_writable(cache)})')
    core = os.getenv('LVMCORE_DIR')
    if core:
        excludes = os.path.join(core, 'etc', 'exclude-exposures.yaml')
        report('  exclude list', excludes if os.path.isfile(excludes) else f'{excludes} (not found)',
               None if os.path.isfile(excludes) else 'no exclude list in LVMCORE_DIR')
    esosky = os.getenv('LVM_ESOSKY_DIR')
    report('  LVM_ESOSKY_DIR', esosky or 'not set (optional)')

    # batch runs
    log.info('batch runs (drp cluster)')
    if slurm_available():
        report('  Slurm', f'available ({shutil.which("sbatch")}): test it with envcheck --slurm')
    else:
        report('  Slurm', 'not available here: batch runs need drp cluster --local (test with envcheck --batch)')
    return problems


def _check_dir(base: str, kind: str) -> str:
    """Returns a new directory for the logs of a batch check, under `base`"""
    path = os.path.abspath(os.path.join(base, f'envcheck-{kind}-{time.strftime("%Y%m%d-%H%M%S")}'))
    os.makedirs(path)
    return path


def _expected_outcomes(log_dir: str) -> bool:
    """Checks summary.txt of a batch check: each dummy job has the outcome it is named after"""
    with open(os.path.join(log_dir, 'summary.txt')) as f:
        lines = [line.split() for line in f if line.strip()]
    outcomes = {words[0]: words[3] for words in lines if len(words) > 3}
    ok = True
    for name, _ in CHECK_JOBS:
        got = outcomes.get(name, 'missing')
        if got == name:
            log.info(f'  job {name:8s} ended as {got}: as expected')
        else:
            log.error(f'  job {name:8s} ended as {got}: expected {name}')
            ok = False
    node = open(os.path.join(log_dir, 'done.log')).read().split('running on ')[-1].strip() \
        if os.path.isfile(os.path.join(log_dir, 'done.log')) else ''
    if node:
        log.info(f'  the jobs ran on {node}')
    return ok


def check_batch_local(base: str = '.') -> bool:
    """Runs the dummy jobs through the local batch code (``drp cluster --local``) and checks them"""
    from lvmdrp.utils.cluster import run_local, report as batch_report, RUN_EXIT_CODES
    log_dir = _check_dir(base, 'batch')
    log.info(f'local batch check: 3 dummy jobs, logs in {log_dir}')
    results = run_local(CHECK_JOBS, np=len(CHECK_JOBS), log_dir=log_dir, stagger=0)
    ok = _expected_outcomes(log_dir) and batch_report(results) == RUN_EXIT_CODES['failed']
    log.info('local batch check passed' if ok else 'local batch check FAILED')
    return ok


def check_batch_slurm(base: str = '.', alloc: str = 'sdss-np', walltime: str = '00:05:00', mem: str = '2G',
                      timeout: float = 15.0, poll: float = 10.0) -> bool:
    """Runs the dummy jobs through Slurm (``sbatch``, as ``drp cluster``), waits and checks them

    The logs go to a new directory under `base`, which must be on a file system the compute
    nodes share (not /tmp). Waits at most `timeout` minutes for the final job to write summary.txt.
    """
    from lvmdrp.utils.cluster import slurm_available, submit_slurm
    if not slurm_available():
        log.error('Slurm is not available on this machine (no sbatch): nothing to check; '
                  'batch runs here need drp cluster --local')
        return False
    log_dir = _check_dir(base, 'slurm')
    log.info(f'Slurm check: 3 dummy jobs with --alloc {alloc}, {walltime} and {mem} each; logs in {log_dir}')
    if not submit_slurm(CHECK_JOBS, log_dir=log_dir, label='lvm_envcheck', alloc=alloc, walltime=walltime, mem=mem,
                        max_running=len(CHECK_JOBS), drpall=False):
        log.error('Slurm check FAILED: the jobs could not be submitted (see above)')
        return False
    summary = os.path.join(log_dir, 'summary.txt')
    start = time.time()
    last_note = start
    while not os.path.isfile(summary):
        if time.time() - start > timeout * 60:
            log.error(f'Slurm check not finished after {timeout:g} minutes: the jobs may still be queued '
                      f'(squeue --me); when they have run, the result is in {summary}')
            return False
        if time.time() - last_note > 60:
            log.info(f'  waiting for the jobs ({(time.time() - start) / 60:.0f} min so far)')
            last_note = time.time()
        time.sleep(poll)
    time.sleep(min(poll, 2))      # let the final job finish writing
    ok = _expected_outcomes(log_dir)
    log.info('Slurm check passed' if ok else 'Slurm check FAILED')
    return ok


def check_environment_variables(argv: list = None) -> None:
    """Entry point of ``envcheck``: reports the environment and, if asked, checks batch runs"""
    parser = argparse.ArgumentParser(prog='envcheck', description=__doc__.split('\n\n')[0])
    parser.add_argument('--batch', action='store_true',
                        help='also run three dummy jobs through the local batch code (drp cluster --local)')
    parser.add_argument('--slurm', action='store_true',
                        help='also run three dummy jobs through Slurm with sbatch (drp cluster) and wait for them')
    parser.add_argument('--log-dir', default='.',
                        help='where the batch checks make their log directory (default: here); with --slurm it '
                             'must be on a file system the compute nodes share')
    parser.add_argument('-a', '--alloc', default='sdss-np', help='with --slurm: the account, partition and qos')
    parser.add_argument('-w', '--walltime', default='00:05:00', help='with --slurm: the time of each job')
    parser.add_argument('--mem', default='2G', help='with --slurm: the memory of each job')
    parser.add_argument('--timeout', type=float, default=15.0, help='with --slurm: minutes to wait for the jobs')
    args = parser.parse_args(argv)

    problems = check_environment()
    missing = [p for p in problems if p.startswith('missing:')]
    if problems:
        log.warning(f'{len(problems)} problem(s) found')
    if missing:
        log.error(f"the following required environment variables are not set: "
                  f"{', '.join(p.split(': ', 1)[1] for p in missing)}")
        log.info("see the README for more information about how to set these variables")
        sys.exit(1)
    log.info("all required environment variables for the LVM DRP are set")

    ok = True
    if args.batch:
        print(file=sys.stderr)          # a blank line, on the log's stream
        log.info('now running the local batch check')
        ok &= check_batch_local(args.log_dir)
    if args.slurm:
        print(file=sys.stderr)          # a blank line, on the log's stream
        log.info('now running the Slurm check')
        ok &= check_batch_slurm(args.log_dir, alloc=args.alloc, walltime=args.walltime, mem=args.mem,
                                timeout=args.timeout)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    check_environment_variables()
