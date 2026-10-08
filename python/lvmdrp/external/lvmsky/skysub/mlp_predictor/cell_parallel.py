"""Fork-based parallel map for diagnostic cell bodies.

Why this exists.  The diagnostic cells are ``exec``'d source, not importable
modules, so a worker defined inside one has no module path and cannot be
pickled by reference -- which rules out ``spawn``, the macOS default.  ``fork``
can do it: the child inherits the parent's whole address space, including the
cell's closures, the hoisted reconstruction basis, the open FITS caches and the
telluric transmission model.  Nothing of that has to be rebuilt or shipped; the
only things crossing the process boundary are the work items going in and the
results coming back.

The pickling problem is solved by a trampoline: the parent stores the callable
in a module global, and the pool is handed :func:`_call_chunk`, which IS
importable and so IS picklable.  The child finds the callable already in its
inherited copy of this module.

Fork has a real failure mode on macOS -- a parent that has threads running
(Accelerate/OpenMP BLAS pools do) can deadlock or crash the child -- so every
call is wrapped and falls back to a serial pass with a printed reason rather
than taking the caller's diagnostic down with it.  ``mlp_predictor.
moon_model_cache`` has forked this way in production since 2026-09, so the
pattern is not new here; the fallback is because *this* caller runs after heavy
numpy work in the parent, which that one does not.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback

_FN = None


def _call_chunk(chunk):
    """Trampoline: run the parent's callable over one chunk of work items."""
    return [(i, _FN(i)) for i in chunk]


def map_indexed(fn, items, n_workers=8, chunk_size=None, verbose=True,
                label="work"):
    """Return ``[fn(i) for i in items]``, computed over a fork pool.

    ``fn`` may be a closure -- that is the point.  Results are reordered back
    into ``items`` order, so the caller sees no difference from a serial loop.

    Falls back to serial, with a printed reason, when ``n_workers <= 1``, when
    the platform has no ``fork``, or when anything at all goes wrong in the
    pool.  A diagnostic that runs slowly is worth far more than one that dies.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    n_workers = int(n_workers)
    reason = None
    if n_workers <= 1:
        reason = "n_workers <= 1"
    elif "fork" not in mp.get_all_start_methods():
        reason = f"no fork start method on {os.name}"
    elif n < 2 * n_workers:
        # Forking 8 children to do 6 items costs more than it saves.
        reason = f"only {n} item(s) for {n_workers} workers"

    if reason is None:
        if chunk_size is None:
            # Aim for ~4 chunks per worker: enough to even out the tail when
            # rows cost different amounts, few enough that each IPC payload
            # stays large relative to its overhead.
            chunk_size = max(1, min(32, n // (4 * n_workers) or 1))
        chunks = [items[i:i + chunk_size] for i in range(0, n, chunk_size)]
        t0 = time.perf_counter()
        if verbose:
            print(f"  {label}: {n} item(s) on {n_workers} fork workers "
                  f"({len(chunks)} chunks of {chunk_size})", flush=True)
        global _FN
        _FN = fn
        try:
            out = {}
            done = 0
            _next_report = max(1, n // 5)
            with mp.get_context("fork").Pool(processes=n_workers) as pool:
                for part in pool.imap_unordered(_call_chunk, chunks):
                    for i, res in part:
                        out[i] = res
                    done += len(part)
                    if verbose and done >= _next_report:
                        el = time.perf_counter() - t0
                        print(f"    {done}/{n}  {el:.0f}s elapsed, "
                              f"~{el / max(done, 1) * (n - done):.0f}s left",
                              flush=True)
                        _next_report += max(1, n // 5)
            if verbose:
                print(f"  {label}: {time.perf_counter() - t0:.1f}s on "
                      f"{n_workers} workers", flush=True)
            return [out[i] for i in items]
        except Exception:
            print(f"  {label}: parallel pass FAILED, falling back to serial.\n"
                  + "".join(traceback.format_exc().splitlines(True)[-3:]),
                  flush=True)
        finally:
            _FN = None
    elif verbose:
        print(f"  {label}: serial ({reason})", flush=True)

    t0 = time.perf_counter()
    res = [fn(i) for i in items]
    if verbose:
        print(f"  {label}: {time.perf_counter() - t0:.1f}s serial", flush=True)
    return res


__all__ = ["map_indexed"]
