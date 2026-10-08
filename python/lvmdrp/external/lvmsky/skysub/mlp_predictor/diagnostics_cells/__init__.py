"""Diagnostic cell bodies — one .py per figure.

Each file is the verbatim body of a diagnostic cell from the split-zodi
notebook; :class:`mlp_predictor.diagnostics.Diagnostics` loads and
execs them against a persistent globals dict.  Do NOT import these
modules directly — the bare names inside each file only resolve inside
the exec context.
"""
