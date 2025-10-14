
#!/usr/bin/env python
# encoding: utf-8

import os
from typing import Union

from lvmdrp import log
from lvmdrp.main import read_expfile, parse_expnums

# set temporary slurm envvars for non-utah runs
os.environ["SLURM_SCRATCH_DIR"] = os.getenv("SLURM_SCRATCH_DIR", "~")
os.environ["SLURM_LOGS_DIR"] = os.getenv("SLURM_LOGS_DIR", "~")

try:
    from slurm import queue
except ImportError:
    queue = None


# TODO: separate cluster run routine for calibrations


def run_cluster(mjds: list = None, expnums: Union[list, str] = None, nodes: int = 2, ppn: int = 64, walltime: str = '24:00:00',
                alloc: str = 'sdss-np', submit: bool = True, run_calibs: str = None, drp_options: str = None, dry_run: bool = False):
    """ Submit a slurm cluster Utah job

    Creates the cluster job at $SLURM_SCRATCH_DIR, e.g /scratch/general/nfs1/[unid]/pbs
    in a sub-directory of the job label, e.g. "lvm_cluster_run", with job id, i.e.
    "/scratch/general/nfs1/[unid]/pbs/[label]/[jobid]".  Logs are available at
    $SLURM_LOGS_DIR.  The cluster adopts the environment from which the cluster job
    was submitted.

    This requires the specific module "slurm/notchpeak-pipelines" to be loaded, and the
    following package versions installed in the Utah miniconda environment:
    flask==2.1.2 flask-sqlalchemy==2.5.1 werkzeug==2.0.1 sqlalchemy==1.4.23

    Parameters
    ----------
    mjds : list, optional
        a list of MJDs to submit to the cluster
    expnums : list|str, optional
        a single exposure number, a range or a file of exposure numbers
    nodes : int, optional
        the number of nodes to use, by default 2
    ppn : int, optional
        the number of CPU cores per node, by default 64
    walltime : str, optional
        the time of which the job is allowed to run, by default '24:00:00'
    alloc : str, optional
        which partition to use, by default 'sdss-np'
    submit : bool, optional
        Flag to submit the job or not, by default True
    run_calibs : str, optional
        Whether to run 'long-term' or 'nightly' calibrations only, by default None (science reduction only)
    drp_options : str, optional
        Pass options to 'drp run' command. See drp run help to see available options
    dry_run : bool, optional
        Logs useful information abaut the current setup without actually running, by default False
    """

    if not queue and not dry_run:
        log.error('No slurm queue module available.  Cannot submit cluster run.')
        return

    # create the slurm queue
    if not dry_run:
        q = queue()
        q.verbose = True
        q.create(label='lvm_cluster_run', nodes=nodes, ppn=ppn, walltime=walltime, alloc=alloc, shared=True)
    else:
        q = []

    cmd = "run"
    if run_calibs is not None:
        expnums = None
        cmd = f"calibrations {run_calibs}"
    else:
        # skip drpall summary file in cluster runs to avoid race condition errors
        drp_options += " --skip-drpall" if "--skip-drpall" not in drp_options else ""

    if expnums is not None:
        if isinstance(expnums, str) and os.path.isfile(expnums):
            expnums = read_expfile(expnums)
        else:
            expnums = parse_expnums(expnums)

        if isinstance(expnums, tuple):
            log.error(f"a closed range of exposure numbers is required, {expnums} given instead.")
            return
        else:
            for expnum in expnums:
                q.append(f"umask 002 && drp run -e {expnum} {drp_options}")
    else:
        # get a list of mjds
        mjds = mjds or sorted(os.listdir(os.getenv('LVM_DATA_S')))
        mjds = list(filter(lambda mjd: mjd.isdigit() if isinstance(mjd, str) else True, mjds))

        for mjd in mjds:
            script = f"umask 002 && drp {cmd} -m {mjd} {drp_options}"
            q.append(script)

    # submit the queue
    if not dry_run:
        q.commit(hard=True, submit=submit)
    else:
        log.info(f"queue for cluster run with {len(q)} nights:")
        for run_ in q:
            log.info(f"   {run_}")
