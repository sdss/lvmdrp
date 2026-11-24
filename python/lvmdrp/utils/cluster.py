#!/usr/bin/env python
# encoding: utf-8
"""Batch runs of the DRP: through the Utah Slurm queue, or locally, N processes at a time

Both kinds of run execute the same list of ``drp`` commands (one per exposure, night or
calibration epoch), built by :func:`science_jobs` or :func:`calibration_jobs`:

- :func:`submit_slurm` puts them in a Slurm queue (the ``slurm`` package and module
  ``slurm/notchpeak-pipelines`` at Utah);
- :func:`run_local` runs them on the current machine, ``np`` at a time, with one log per job
  and one line per finished job in a summary file.
"""

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union

from lvmdrp import log
from lvmdrp.main import read_expfile, parse_expnums, RUN_EXIT_CODES

# set temporary slurm envvars for non-utah runs
os.environ["SLURM_SCRATCH_DIR"] = os.getenv("SLURM_SCRATCH_DIR", "~")
os.environ["SLURM_LOGS_DIR"] = os.getenv("SLURM_LOGS_DIR", "~")

try:
    from slurm import queue # type: ignore
except ImportError:
    queue = None


#: thread limits for each local job: the DRP steps are single-process, and many run at once
LOCAL_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')

#: calibration routines of ``drp calibrations`` that take --from-epochs
FROM_EPOCHS_ROUTINES = {'long-term'}


def _mjd_list(mjds) -> list:
    """Returns a list of integer MJDs from one MJD or a list, by default all MJDs in $LVM_DATA_S"""
    if mjds is None or (isinstance(mjds, (list, tuple)) and len(mjds) == 0):
        return _all_mjds()
    if not isinstance(mjds, (list, tuple)):
        mjds = [mjds]
    return [int(mjd) for mjd in mjds if str(mjd).isdigit()]


def _all_mjds() -> list:
    """Returns the MJDs of the raw data directory, $LVM_DATA_S"""
    return sorted(int(mjd) for mjd in os.listdir(os.getenv('LVM_DATA_S')) if mjd.isdigit())


def _expnum_list(expnums: Union[int, str, list, tuple]) -> Union[list, None]:
    """Returns a list of exposure numbers from a number, a closed range, a list or a file"""
    if isinstance(expnums, str) and os.path.isfile(expnums):
        return read_expfile(expnums)
    expnums = parse_expnums(expnums)
    if isinstance(expnums, tuple):
        log.error(f"a closed range of exposure numbers is required, {expnums} given instead.")
        return None
    return expnums


def science_jobs(mjds: list = None, expnums: Union[int, str, list] = None,
                 drp_options: str = '') -> List[Tuple[str, str]]:
    """Builds the ``drp run`` commands of a batch science reduction

    One command per exposure if `expnums` is given, else one per night. ``--skip-drpall`` is
    always added, because jobs running at the same time would write the same drpall file;
    the drpall is made once at the end (``drp summary``).

    Parameters
    ----------
    mjds : list, optional
        the nights to reduce, by default all nights in $LVM_DATA_S (when `expnums` is not given)
    expnums : int|str|list, optional
        an exposure number, a closed range ("19475-19514"), a list, or a file of exposure numbers
    drp_options : str, optional
        options passed to every ``drp run``

    Returns
    -------
    list[tuple[str, str]]
        (name, command) per job; the name is the exposure number or the MJD
    """
    drp_options = (drp_options or '').strip()
    if '--skip-drpall' not in drp_options.split() and '-da' not in drp_options.split():
        drp_options = f'{drp_options} --skip-drpall'.strip()

    if expnums is not None:
        expnums = _expnum_list(expnums)
        if expnums is None:
            return []
        return [(str(expnum), f"umask 002 && drp run -e {expnum} {drp_options}") for expnum in expnums]

    mjds = _mjd_list(mjds)
    return [(str(mjd), f"umask 002 && drp run -m {mjd} {drp_options}") for mjd in mjds]


def calibration_jobs(mjds: list, routine: str = 'long-term', from_epochs: bool = False,
                     drp_options: str = '') -> List[Tuple[str, str]]:
    """Builds the ``drp calibrations <routine>`` commands of a batch calibration run, one per MJD

    Parameters
    ----------
    mjds : list
        the MJDs (calibration epochs, or nights) to process
    routine : str, optional
        the ``drp calibrations`` routine, by default 'long-term'
    from_epochs : bool, optional
        add --from-epochs (only for the routines that take it, see ``FROM_EPOCHS_ROUTINES``)
    drp_options : str, optional
        options passed to every command

    Returns
    -------
    list[tuple[str, str]]
        (name, command) per job; the name is '<routine>-<mjd>'
    """
    drp_options = (drp_options or '').strip()
    flag = ' --from-epochs' if from_epochs and routine in FROM_EPOCHS_ROUTINES else ''
    if from_epochs and not flag:
        log.warning(f"drp calibrations {routine} has no --from-epochs option; the MJDs are passed with -m")
    jobs = []
    for mjd in mjds:
        command = f"umask 002 && drp calibrations {routine}{flag} -m {mjd} {drp_options}".strip()
        jobs.append((f'{routine}-{mjd}', command))
    return jobs


def nights_of(expnums: list) -> list:
    """Returns the sorted nights (MJDs) of a list of exposure numbers, from the raw data"""
    from lvmdrp.utils.paths import mjd_from_expnum
    nights = set()
    for expnum in expnums:
        try:
            nights.add(mjd_from_expnum(int(expnum))[0])
        except ValueError as err:
            log.warning(f"{err}")
    return sorted(nights)


def regenerate_metadata(mjds: list) -> None:
    """Regenerates the frames metadata of each night, one at a time

    Done once before the jobs of a batch start, so that jobs reducing exposures of the same
    night don't all build that night's metadata at the same time (sdss/lvmdrp#235).
    """
    from lvmdrp.utils.metadata import get_frames_metadata
    for mjd in mjds:
        log.info(f"regenerating the metadata of MJD {mjd}")
        get_frames_metadata(mjd=int(mjd), overwrite=True)


def submit_slurm(jobs: List[Tuple[str, str]], label: str = 'lvm_cluster_run', nodes: int = 2, ppn: int = 64,
                 walltime: str = '24:00:00', alloc: str = 'sdss-np', submit: bool = True) -> bool:
    """Puts the commands of a batch run in a Slurm queue

    The job is created at $SLURM_SCRATCH_DIR, e.g /scratch/general/nfs1/[unid]/pbs, in a
    sub-directory named by `label` and the job id; logs are at $SLURM_LOGS_DIR. The cluster
    adopts the environment from which the job was submitted. This requires the module
    "slurm/notchpeak-pipelines" and the ``slurm`` package.

    Returns
    -------
    bool
        False if no Slurm queue is available
    """
    if not queue:
        log.error('No slurm queue module available. Cannot submit cluster run.')
        return False
    q = queue()
    q.verbose = True
    q.create(label=label, nodes=nodes, ppn=ppn, walltime=walltime, alloc=alloc, shared=True)
    for _, command in jobs:
        q.append(command)
    q.commit(hard=True, submit=submit)
    return True


def outcome(exit_code: int) -> str:
    """Returns 'done', 'skipped' or 'failed' for the exit code of a ``drp`` command"""
    if exit_code == RUN_EXIT_CODES['done']:
        return 'done'
    if exit_code == RUN_EXIT_CODES['skipped']:
        return 'skipped'
    return 'failed'


def run_local(jobs: List[Tuple[str, str]], np: int = 30, log_dir: str = 'logs', stagger: float = 20.0,
              summary_file: str = None) -> List[dict]:
    """Runs the commands of a batch run on this machine, `np` at a time

    Each command runs in its own process (``bash -c``), with its output in
    ``<log_dir>/<name>.log`` and the thread limits of ``LOCAL_THREAD_ENV`` set to 1 unless
    already set. The first `np` jobs start `stagger` seconds apart, so that they don't all read
    the same calibration files at once; later jobs start as earlier ones end. One line per
    finished job (name, exit code, outcome, minutes, time) is appended to `summary_file`.

    Parameters
    ----------
    jobs : list[tuple[str, str]]
        (name, command) per job
    np : int, optional
        the number of jobs running at once, by default 30
    log_dir : str, optional
        directory for the job logs, by default 'logs' (in the current directory)
    stagger : float, optional
        seconds between the starts of the first `np` jobs, by default 20
    summary_file : str, optional
        file for one line per finished job, by default <log_dir>/summary.txt

    Returns
    -------
    list[dict]
        per job, in the order given: name, command, exit, outcome, minutes, log
    """
    os.makedirs(log_dir, exist_ok=True)
    summary_file = summary_file or os.path.join(log_dir, 'summary.txt')
    env = dict(os.environ)
    for name in LOCAL_THREAD_ENV:
        env.setdefault(name, '1')
    lock = threading.Lock()
    np = max(1, int(np))

    def _run(index, name, command):
        if index < np and stagger > 0:
            time.sleep(index * stagger)
        log_path = os.path.join(log_dir, f'{name}.log')
        start = time.time()
        with open(log_path, 'w') as logfile:
            logfile.write(f'# {command}\n')
            logfile.flush()
            exit_code = subprocess.run(['bash', '-c', command], stdout=logfile, stderr=subprocess.STDOUT,
                                       env=env).returncode
        result = {'name': name, 'command': command, 'exit': exit_code, 'outcome': outcome(exit_code),
                  'minutes': (time.time() - start) / 60, 'log': log_path}
        line = (f"{name} exit {exit_code} {result['outcome']} {result['minutes']:.1f} min "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
        with lock:
            with open(summary_file, 'a') as summary:
                summary.write(line + '\n')
            log.info(line)
        return result

    log.info(f"running {len(jobs)} jobs, {np} at a time; logs in {log_dir}, one line per job in {summary_file}")
    with ThreadPoolExecutor(max_workers=np) as pool:
        futures = [pool.submit(_run, i, name, command) for i, (name, command) in enumerate(jobs)]
        results = [future.result() for future in futures]
    return results


def report(results: List[dict]) -> int:
    """Logs the outcome of a local batch run and returns its exit code (as ``drp run``)"""
    groups = {kind: [r['name'] for r in results if r['outcome'] == kind] for kind in ('done', 'skipped', 'failed')}
    parts = [f'{len(groups[k])} {k}' + (f' ({", ".join(groups[k])})' if k != 'done' and groups[k] else '')
             for k in ('done', 'skipped', 'failed')]
    log.info(f"batch run: {'; '.join(parts)}")
    if groups['failed']:
        return RUN_EXIT_CODES['failed']
    if groups['skipped']:
        return RUN_EXIT_CODES['skipped']
    return RUN_EXIT_CODES['done']


def run_cluster(mjds: list = None, expnums: Union[list, str] = None, nodes: int = 2, ppn: int = 64, walltime: str = '24:00:00',
                alloc: str = 'sdss-np', submit: bool = True, drp_options: str = None, dry_run: bool = False,
                local: bool = False, np: int = 30, log_dir: str = 'logs', stagger: float = 20.0,
                metadata: bool = True, summary: bool = True) -> int:
    """ Batch run of the DRP science reduction, through Slurm or locally

    Builds one ``drp run`` per exposure (or per night, without `expnums`), always with
    ``--skip-drpall``. With exposures, the metadata of their nights is regenerated first, one
    night at a time (#235). Then the jobs go to a Slurm queue, or, with `local`, run on this
    machine `np` at a time (:func:`run_local`), after which the drpall is made once (``drp summary``).

    Parameters
    ----------
    mjds : list, optional
        a list of MJDs to reduce
    expnums : list|str, optional
        a single exposure number, a range or a file of exposure numbers
    nodes : int, optional
        Slurm: the number of nodes to use, by default 2
    ppn : int, optional
        Slurm: the number of CPU cores per node, by default 64
    walltime : str, optional
        Slurm: the time of which the job is allowed to run, by default '24:00:00'
    alloc : str, optional
        Slurm: which partition to use, by default 'sdss-np'
    submit : bool, optional
        Slurm: flag to submit the job or not, by default True
    drp_options : str, optional
        Pass options to 'drp run' command. See drp run help to see available options
    dry_run : bool, optional
        Logs the jobs without running or submitting them, by default False
    local : bool, optional
        run the jobs on this machine instead of submitting them to Slurm, by default False
    np : int, optional
        local: the number of jobs at once, by default 30
    log_dir : str, optional
        local: directory of the job logs and summary, by default 'logs'
    stagger : float, optional
        local: seconds between the starts of the first `np` jobs, by default 20
    metadata : bool, optional
        regenerate the metadata of the exposures' nights first, by default True
    summary : bool, optional
        local: make the drpall once the jobs have ended, by default True

    Returns
    -------
    int
        exit code: as ``drp run`` for a local run (0, 3 if some skipped, 1 if any failed);
        0 once submitted to Slurm; 1 if nothing could be run or submitted
    """
    jobs = science_jobs(mjds=mjds, expnums=expnums, drp_options=drp_options)
    if not jobs:
        log.error("no jobs to run")
        return 1
    mode = f"locally, {np} at a time" if local else f"through Slurm ({nodes} nodes x {ppn} cores, {alloc})"
    nights = nights_of([name for name, _ in jobs]) if expnums is not None and metadata else []

    if dry_run:
        log.info(f"would run {len(jobs)} jobs {mode}:")
        for _, command in jobs:
            log.info(f"   {command}")
        if nights:
            log.info(f"after regenerating the metadata of MJDs {nights}")
        if local and summary:
            log.info("and then make the drpall (drp summary)")
        return 0

    if not local and not queue:
        log.error('No slurm queue module available. Cannot submit cluster run (use --local to run here).')
        return 1
    regenerate_metadata(nights)

    if not local:
        submit_slurm(jobs, label='lvm_cluster_run', nodes=nodes, ppn=ppn, walltime=walltime, alloc=alloc, submit=submit)
        log.info("when the jobs have ended, make the drpall with: drp summary")
        return 0

    results = run_local(jobs, np=np, log_dir=log_dir, stagger=stagger)
    code = report(results)
    if summary and any(r['outcome'] == 'done' for r in results):
        from lvmdrp.main import create_drpall
        log.info("making the drpall summary file")
        create_drpall()
    return code


def run_cluster_cals(mjds: list = None, from_epochs: bool = True, nodes: int = 2, ppn: int = 64, walltime: str = '24:00:00',
                     alloc: str = 'sdss-np', submit: bool = True, routine: str = "long-term", drp_options: str = None,
                     dry_run: bool = False, local: bool = False, np: int = 30, log_dir: str = 'logs',
                     stagger: float = 20.0) -> int:
    """ Batch run of a calibration routine, one job per MJD, through Slurm or locally

    See :func:`run_cluster` for the Slurm and local options.

    Returns
    -------
    int
        0 once submitted, or when every local job ended with exit 0; 1 otherwise
    """
    mjds = _mjd_list(mjds)
    jobs = calibration_jobs(mjds, routine=routine, from_epochs=from_epochs, drp_options=drp_options)
    if not jobs:
        log.error("no jobs to run")
        return 1
    mode = f"locally, {np} at a time" if local else f"through Slurm ({nodes} nodes x {ppn} cores, {alloc})"

    if dry_run:
        log.info(f"would run {len(jobs)} calibration jobs {mode}:")
        for _, command in jobs:
            log.info(f"   {command}")
        return 0

    if not local:
        return 0 if submit_slurm(jobs, label='lvm_cals', nodes=nodes, ppn=ppn, walltime=walltime, alloc=alloc,
                                 submit=submit) else 1

    results = run_local(jobs, np=np, log_dir=log_dir, stagger=stagger)
    report(results)
    return 0 if all(r['exit'] == 0 for r in results) else 1
