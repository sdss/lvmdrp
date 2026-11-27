#!/usr/bin/env python
# encoding: utf-8
"""Batch runs of the DRP: through Slurm (``sbatch``), or locally, N processes at a time

Both kinds of run execute the same list of ``drp`` commands (one per exposure, night or
calibration epoch), built by :func:`science_jobs` or :func:`calibration_jobs`:

- :func:`submit_slurm` submits them with ``sbatch`` as a Slurm job array, one task per command,
  followed by a job that collects the outcomes (:func:`finish_slurm`); nothing Slurm-related
  is needed in the Python environment (sdss/lvmdrp#285);
- :func:`run_local` runs them on the current machine, ``np`` at a time.

Either way each job has its log in the log directory, and ``summary.txt`` there gets one line
per job (name, exit code, outcome, minutes, time).
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union

from lvmdrp import log
from lvmdrp.main import read_expfile, parse_expnums, RUN_EXIT_CODES


#: thread limits for each job: the DRP steps are single-process, and many run at once
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


def slurm_available() -> bool:
    """True if ``sbatch`` can be found, i.e. jobs can be submitted to Slurm from here"""
    return shutil.which('sbatch') is not None


def _sbatch_header(label: str, alloc: str, walltime: str, mem: str, output: str) -> List[str]:
    """Returns the #SBATCH lines of a one-core job; `alloc` names the account, partition and qos"""
    return ['#!/bin/bash',
            f'#SBATCH --job-name={label}',
            f'#SBATCH --account={alloc}',
            f'#SBATCH --partition={alloc}',
            f'#SBATCH --qos={alloc}',
            f'#SBATCH --time={walltime}',
            '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1',
            f'#SBATCH --mem={mem}',
            f'#SBATCH --output={output}']


#: body of each task of the job array: runs line $SLURM_ARRAY_TASK_ID of the job list, with its
#: output in <log dir>/<name>.log, and writes its summary line to <log dir>/status/<name>
_TASK_BODY = r"""
for var in @THREADS@; do export "$var=${!var:-1}"; done
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" '@JOBS@')
name=${line%%$'\t'*}
command=${line#*$'\t'}
log='@LOGDIR@'/"$name.log"
start=$(date +%s)
echo "# $command" > "$log"
bash -c "$command" >> "$log" 2>&1
code=$?
case $code in
    @DONE@) outcome=done ;;
    @SKIPPED@) outcome=skipped ;;
    *) outcome=failed ;;
esac
minutes=$(awk -v s="$start" -v e="$(date +%s)" 'BEGIN { printf "%.1f", (e - s) / 60 }')
echo "$name exit $code $outcome $minutes min $(date '+%Y-%m-%d %H:%M:%S')" > '@LOGDIR@'/status/"$name"
exit $code
"""


def write_slurm_scripts(jobs: List[Tuple[str, str]], log_dir: str = 'logs', label: str = 'lvm_cluster_run',
                        alloc: str = 'sdss-np', walltime: str = '24:00:00', mem: str = '16G',
                        max_running: int = 128, drpall: bool = False) -> Tuple[str, str]:
    """Writes the job list, the Slurm job-array script and the script of the final job, in `log_dir`

    - ``jobs.txt``: one line per job, name and command separated by a tab;
    - ``<label>.sbatch``: a job array with one one-core task per line of ``jobs.txt``, at most
      `max_running` at a time, each with the `walltime` and `mem` given;
    - ``<label>-finish.sbatch``: run after the array has ended (:func:`finish_slurm`): writes
      ``summary.txt``, reports the outcomes and, with `drpall`, makes the drpall.

    Jobs run in the environment ``sbatch`` was called from, so the scripts need no setup.

    Returns
    -------
    tuple[str, str]
        the paths of the array script and of the final script
    """
    log_dir = os.path.abspath(log_dir)
    os.makedirs(os.path.join(log_dir, 'status'), exist_ok=True)
    jobs_file = os.path.join(log_dir, 'jobs.txt')
    with open(jobs_file, 'w') as f:
        for name, command in jobs:
            f.write(f'{name}\t{command}\n')

    array_script = os.path.join(log_dir, f'{label}.sbatch')
    lines = _sbatch_header(label, alloc, walltime, mem, os.path.join(log_dir, 'slurm-%A_%a.out'))
    lines.append(f'#SBATCH --array=1-{len(jobs)}%{max(1, int(max_running))}')
    body = (_TASK_BODY.replace('@THREADS@', ' '.join(LOCAL_THREAD_ENV)).replace('@JOBS@', jobs_file)
            .replace('@LOGDIR@', log_dir).replace('@DONE@', str(RUN_EXIT_CODES['done']))
            .replace('@SKIPPED@', str(RUN_EXIT_CODES['skipped'])))
    with open(array_script, 'w') as f:
        f.write('\n'.join(lines) + '\n' + body)

    finish_script = os.path.join(log_dir, f'{label}-finish.sbatch')
    lines = _sbatch_header(f'{label}-finish', alloc, walltime, mem, os.path.join(log_dir, 'slurm-finish-%j.out'))
    command = f"python -m lvmdrp.utils.cluster finish '{log_dir}'" + (' --drpall' if drpall else '')
    with open(finish_script, 'w') as f:
        f.write('\n'.join(lines) + '\n\n' + command + '\n')
    return array_script, finish_script


def _sbatch(args: List[str]) -> Union[str, None]:
    """Runs ``sbatch --parsable <args>`` and returns the job id, or None (logged) if it failed"""
    result = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"sbatch {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
        return None
    return result.stdout.strip().split(';')[0]


def submit_slurm(jobs: List[Tuple[str, str]], log_dir: str = 'logs', label: str = 'lvm_cluster_run',
                 alloc: str = 'sdss-np', walltime: str = '24:00:00', mem: str = '16G', max_running: int = 128,
                 drpall: bool = False, submit: bool = True) -> bool:
    """Submits the commands of a batch run to Slurm with ``sbatch``, as a job array

    Writes the scripts (:func:`write_slurm_scripts`), submits the array, then the final job with
    ``--dependency=afterany`` on it, so that it runs once every task has ended, however it ended.

    Returns
    -------
    bool
        False if ``sbatch`` is not available or a submission failed
    """
    if not slurm_available():
        log.error('sbatch not found: Slurm is not available here (use --local to run on this machine)')
        return False
    array_script, finish_script = write_slurm_scripts(jobs, log_dir=log_dir, label=label, alloc=alloc,
                                                      walltime=walltime, mem=mem, max_running=max_running,
                                                      drpall=drpall)
    if not submit:
        log.info(f"scripts written, not submitted: sbatch {array_script}, then "
                 f"sbatch --dependency=afterany:<job id> {finish_script}")
        return True
    array_id = _sbatch([array_script])
    if array_id is None:
        return False
    finish_id = _sbatch([f'--dependency=afterany:{array_id}', finish_script])
    if finish_id is None:
        log.error(f"job array {array_id} was submitted, but not the final job; when the array has ended, "
                  f"run: python -m lvmdrp.utils.cluster finish {os.path.abspath(log_dir)}" + (' --drpall' if drpall else ''))
        return False
    log.info(f"submitted job array {array_id}: {len(jobs)} jobs, at most {max_running} at a time, "
             f"logs in {os.path.abspath(log_dir)}; then job {finish_id} writes summary.txt"
             + (" and makes the drpall" if drpall else ""))
    log.info(f"follow with: squeue --me; when job {finish_id} has ended, see {os.path.abspath(log_dir)}/summary.txt")
    return True


def finish_slurm(log_dir: str, drpall: bool = False) -> int:
    """Collects the outcomes of a Slurm batch run, after its job array has ended

    Reads the job list and the status line of each task (``status/<name>``), writes them to
    ``summary.txt`` in the order of the job list, reports them and, with `drpall`, makes the
    drpall if any job is done. A task without a status line (killed, e.g. at its time or memory
    limit) counts as failed; its Slurm output is ``slurm-<array id>_<task>.out``.

    Returns
    -------
    int
        exit code as ``drp run``: 0 all done, 3 some skipped, 1 any failed
    """
    with open(os.path.join(log_dir, 'jobs.txt')) as f:
        names = [line.split('\t', 1)[0] for line in f if line.strip()]
    results, lines = [], []
    for index, name in enumerate(names, start=1):
        status = os.path.join(log_dir, 'status', name)
        if os.path.isfile(status):
            line = open(status).read().strip()
            exit_code = int(line.split()[2])
            results.append({'name': name, 'exit': exit_code, 'outcome': outcome(exit_code)})
        else:
            line = f'{name} exit none failed (no status: killed? see slurm-*_{index}.out)'
            results.append({'name': name, 'exit': None, 'outcome': 'failed'})
        lines.append(line)
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    code = report(results)
    if drpall and any(r['outcome'] == 'done' for r in results):
        from lvmdrp.main import create_drpall
        log.info("making the drpall summary file")
        create_drpall()
    return code


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
    """Logs the outcome of a batch run and returns its exit code (as ``drp run``)"""
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
                metadata: bool = True, summary: bool = True, mem: str = '16G') -> int:
    """ Batch run of the DRP science reduction, through Slurm or locally

    Builds one ``drp run`` per exposure (or per night, without `expnums`), always with
    ``--skip-drpall``. With exposures, the metadata of their nights is regenerated first, one
    night at a time (#235). Then the jobs are submitted to Slurm as a job array
    (:func:`submit_slurm`), or, with `local`, run on this machine `np` at a time
    (:func:`run_local`). Once all jobs have ended the drpall is made once: at the end of a local
    run, or by the final Slurm job.

    Parameters
    ----------
    mjds : list, optional
        a list of MJDs to reduce
    expnums : list|str, optional
        a single exposure number, a range or a file of exposure numbers
    nodes : int, optional
        Slurm: with `ppn`, at most `nodes` x `ppn` jobs run at once, by default 2
    ppn : int, optional
        Slurm: see `nodes`, by default 64
    walltime : str, optional
        Slurm: the time each job (exposure or night) is allowed to run, by default '24:00:00'
    alloc : str, optional
        Slurm: the account, partition and qos, by default 'sdss-np'
    submit : bool, optional
        Slurm: submit the scripts, or only write them, by default True
    drp_options : str, optional
        Pass options to 'drp run' command. See drp run help to see available options
    dry_run : bool, optional
        Logs the jobs without running or submitting them, by default False
    local : bool, optional
        run the jobs on this machine instead of submitting them to Slurm, by default False
    np : int, optional
        local: the number of jobs at once, by default 30
    log_dir : str, optional
        directory of the job logs and summary.txt (and of the Slurm scripts), by default 'logs'
    stagger : float, optional
        local: seconds between the starts of the first `np` jobs, by default 20
    metadata : bool, optional
        regenerate the metadata of the exposures' nights first, by default True
    summary : bool, optional
        make the drpall once the jobs have ended, by default True
    mem : str, optional
        Slurm: the memory of each job, by default '16G'

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
    max_running = nodes * ppn
    mode = (f"locally, {np} at a time" if local
            else f"through Slurm ({alloc}, at most {max_running} at a time, {walltime} and {mem} each)")
    nights = nights_of([name for name, _ in jobs]) if expnums is not None and metadata else []

    if dry_run:
        log.info(f"would run {len(jobs)} jobs {mode}:")
        for _, command in jobs:
            log.info(f"   {command}")
        if nights:
            log.info(f"after regenerating the metadata of MJDs {nights}")
        if summary:
            log.info("and then make the drpall")
        return 0

    if not local and not slurm_available():
        log.error('sbatch not found: Slurm is not available here (use --local to run on this machine)')
        return 1
    regenerate_metadata(nights)

    if not local:
        ok = submit_slurm(jobs, log_dir=log_dir, label='lvm_cluster_run', alloc=alloc, walltime=walltime,
                          mem=mem, max_running=max_running, drpall=summary, submit=submit)
        return 0 if ok else 1

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
                     stagger: float = 20.0, mem: str = '16G') -> int:
    """ Batch run of a calibration routine, one job per MJD, through Slurm or locally

    See :func:`run_cluster` for the Slurm and local options; no drpall is made.

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
    max_running = nodes * ppn
    mode = (f"locally, {np} at a time" if local
            else f"through Slurm ({alloc}, at most {max_running} at a time, {walltime} and {mem} each)")

    if dry_run:
        log.info(f"would run {len(jobs)} calibration jobs {mode}:")
        for _, command in jobs:
            log.info(f"   {command}")
        return 0

    if not local:
        return 0 if submit_slurm(jobs, log_dir=log_dir, label='lvm_cals', alloc=alloc, walltime=walltime, mem=mem,
                                 max_running=max_running, drpall=False, submit=submit) else 1

    results = run_local(jobs, np=np, log_dir=log_dir, stagger=stagger)
    report(results)
    return 0 if all(r['exit'] == 0 for r in results) else 1


def main(argv: List[str] = None) -> int:
    """Entry point of the final job of a Slurm batch run: ``python -m lvmdrp.utils.cluster finish <log dir>``

    Exits with 0 once summary.txt is written (and the drpall made, if asked), whatever the jobs'
    outcomes, so that Slurm shows the final job as failed only if it could not do its own work;
    the outcomes are in summary.txt and its log.
    """
    parser = argparse.ArgumentParser(prog='python -m lvmdrp.utils.cluster')
    sub = parser.add_subparsers(dest='action', required=True)
    finish = sub.add_parser('finish', help='collect the outcomes of a Slurm batch run (see finish_slurm)')
    finish.add_argument('log_dir', help='the log directory of the run')
    finish.add_argument('--drpall', action='store_true', help='make the drpall if any job is done')
    args = parser.parse_args(argv)
    finish_slurm(args.log_dir, drpall=args.drpall)
    return 0


if __name__ == '__main__':
    sys.exit(main())
