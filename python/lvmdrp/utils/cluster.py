
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



def run_cluster(mjds: list = None, expnums: Union[list, str] = None, nodes: int = 2, ppn: int = 64, walltime: str = '24:00:00',
                alloc: str = 'sdss-np', submit: bool = True):
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
    """

    if not queue:
        log.error('No slurm queue module available.  Cannot submit cluster run.')
        return

    # create the slurm queue
    q = queue()
    q.verbose = True
    q.create(label='lvm_cluster_run', nodes=nodes, ppn=ppn, walltime=walltime, alloc=alloc, shared=True)

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
                q.append(f"drp run -e {expnum} -c")
    else:
        # get a list of mjds
        mjds = mjds or sorted(os.listdir(os.getenv('LVM_DATA_S')))

        for mjd in mjds:
            script = f"drp run -m {mjd} -c"
            q.append(script)

    # submit the queue
    q.commit(hard=True, submit=submit)
