Running the DRP at Utah
=======================

This page is for those who reduce many exposures, or whole epochs, on the Utah (CHPC)
machines. ``drp cluster`` runs one ``drp run`` per exposure or night, either through the
Slurm cluster or locally on one machine. The options it shares with local runs, the log
directory and the exit codes are described under Batch Runs in :doc:`Overview`.

Setting Up
----------

Use your own lvmdrp environment, the same one you use for ``drp run`` (see
:doc:`Installation`). Slurm jobs run in the environment they are submitted from, so
nothing else needs to be installed or loaded: activate the environment, set
``LVMDRP_VERSION`` so that the reductions go to a directory of your own, and check the
setup::

    export LVMDRP_VERSION=<label>
    envcheck

``envcheck`` shows where the reductions will be written and whether that directory can be
written, and ends by saying whether Slurm is available on the machine. Then check that jobs
can be submitted and run, with three dummy jobs that end as done, skipped and failed (the DRP
itself is not run)::

    envcheck --slurm

It waits for the jobs, usually a minute or two, and ends with "Slurm check passed". Run it
from a directory the compute nodes can see (your home or work area, not ``/tmp``), or give
one with ``--log-dir``. It takes the ``-a``, ``-w`` and ``--mem`` options described below.

Slurm or One Machine
--------------------

- **Through Slurm** (``drp cluster`` without ``--local``): the jobs go to the SDSS nodes of
  the cluster, as many at a time as are free and allowed. Best for large runs.
- **On one machine** (``drp cluster --local --np N``): the jobs run ``N`` at a time on the
  machine you are logged in to, for example notchpeak20, which is not part of the Slurm
  cluster. Simple for an epoch or less, and ``drp cluster`` waits and exits with the outcome.

Both run the same ``drp run`` commands and leave the same logs and ``summary.txt``.

Submitting a Run
----------------

::

    drp cluster -F <expnum_file> -w 01:00:00 --log-dir <log_dir> --drp-options "<options for drp run>"

``drp cluster`` submits the jobs with ``sbatch`` as a Slurm job array, one task per job with
one core each, followed by a final job that runs once every task has ended, however it ended.
The options for Slurm:

- ``-a``, ``--alloc``: the Slurm account, partition and qos, ``sdss-np`` (the SDSS notchpeak
  nodes) by default, or ``sdss-kp`` (kingspeak). The ones you can use are listed by::

      sacctmgr -n -P show assoc user=$USER format=account,partition,qos

- ``-w``, ``--walltime``: the time each job may run, 24 hours by default; a job that runs
  longer is stopped. Set it with a margin; a shorter time can start sooner.
- ``--mem``: the memory of each job, ``16G`` by default.
- ``-n``, ``--nodes`` and ``-p``, ``--ppn``: at most nodes x ppn jobs run at once
  (2 x 64 by default).
- ``--dry-run``: list the jobs without submitting anything.

``drp cluster`` returns as soon as the jobs are submitted, with exit code 0 (1 if ``sbatch``
is not found or a submission failed), and logs the two job ids. For large runs, tell the other
users of the SDSS nodes beforehand on the SDSS CHPC users' mailing list.

Several calibration epochs are run in the same way with ``drp calibrations cluster`` (see
:doc:`Calibration`); no drpall is made for them.

Following a Run
---------------

``squeue --me`` shows the tasks of the array (``<array id>_<task>``) and the final job. A
task waiting as ``PD`` with the reason ``(Priority)`` or ``(Resources)`` starts when cores are
free; the final job waits as ``(Dependency)`` until the array has ended. When the final job
has gone from the list, the run is over::

    cat <log_dir>/summary.txt
    tail <log_dir>/slurm-finish-*.out

The final job writes ``summary.txt`` (one line per job: name, exit code, outcome, minutes,
time), logs the number of jobs done, skipped and failed, and makes the drpall (unless
``--skip-summary``). It ends with exit code 0 once it has done this, whatever the outcomes of
the jobs, so ``sacct`` shows it as failed only if it could not write ``summary.txt`` or make
the drpall. In ``sacct`` the tasks of skipped (exit 3) and failed (exit 1) jobs show as
``FAILED``, since Slurm counts every non-zero exit code as a failure; ``summary.txt`` tells them
apart.

The log directory holds, besides ``<name>.log`` for each job and ``summary.txt``:

- ``jobs.txt``: the commands, one per line;
- ``lvm_cluster_run.sbatch`` and ``lvm_cluster_run-finish.sbatch``: the job scripts;
- ``status/``: one line per finished task, collected into ``summary.txt`` by the final job;
- ``slurm-<array id>_<task>.out``: what Slurm itself reports for each task;
- ``slurm-finish-<job id>.out``: the output of the final job.

A Small Test Run
----------------

Before a large run, try a few exposures in a file, plus an exposure number that does not
exist (it should end as failed), with the dry run first::

    drp cluster -F <expnum_file> -n 1 -p 4 -w 01:00:00 --log-dir logs_test --dry-run
    drp cluster -F <expnum_file> -n 1 -p 4 -w 01:00:00 --log-dir logs_test
    squeue --me
    cat logs_test/summary.txt
    tail logs_test/slurm-finish-*.out

Troubleshooting
---------------

- ``summary.txt`` lists a job as ``exit none failed (no status ...)``: Slurm stopped the task.
  Its ``slurm-<array id>_<task>.out`` says why (time limit or memory); raise ``-w`` or ``--mem``.
- ``sbatch: error: Invalid account or account/partition combination``: give an account you
  can use with ``-a`` (see ``sacctmgr`` above).
- "sbatch not found": this machine has no Slurm; use ``--local``, or log in to a machine of
  the cluster.
- No drpall at the end: see the output of the final job, ``slurm-finish-<job id>.out``; it can
  be made by hand with ``drp summary``.
