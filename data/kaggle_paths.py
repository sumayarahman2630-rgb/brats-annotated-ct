"""Shared (Stage 2 + Stage 3) -- auto-corrects the unstable username segment of Kaggle dataset mount paths; output: the real, existing directory to use."""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_DATASETS_PREFIX = "/kaggle/input/datasets/"


def resolve_kaggle_dataset_path(path: str) -> str:
    """Confirmed real, recurring behavior on Kaggle (2026-07-29/30, this
    exact project): a dataset attached via "Add Data" mounts at
    /kaggle/input/datasets/<username>/<slug>/..., but the <username>
    segment is NOT a stable property of the dataset -- it changed between
    two sessions attaching the SAME dataset (sumayarahman30 one session,
    sumayarahmanmeherin the next), breaking a hardcoded config path each
    time. This has nothing to do with data/loaders_synthetic_ct.py's
    already-documented "unusually deep path" (the *rest* of the path,
    after <slug>/, which reflects how the dataset was originally exported
    and genuinely doesn't change) -- only this one path segment is the
    problem.

    If `path` already exists, it's returned unchanged (the common case,
    and the only case that costs nothing extra). Otherwise, if `path`
    looks like /kaggle/input/datasets/<username>/<rest...>, every OTHER
    username actually present under /kaggle/input/datasets/ is tried in
    its place; if exactly one substituted candidate exists, that's
    returned (with a log message, so a silent path swap is never actually
    silent). Zero or multiple matches raise FileNotFoundError rather than
    guessing -- an ambiguous or genuinely-missing dataset needs a human
    decision, not an automatic pick.
    """
    if os.path.isdir(path):
        return path

    if not path.startswith(_DATASETS_PREFIX):
        return path  # not a datasets/ path at all -- nothing this function knows how to fix

    rest = path[len(_DATASETS_PREFIX):]
    parts = rest.split("/", 1)
    if len(parts) < 2:
        return path  # too short to contain a <username>/<slug-and-beyond> split
    stale_username, remainder = parts

    if not os.path.isdir(_DATASETS_PREFIX):
        return path  # /kaggle/input/datasets/ itself doesn't exist here -- let the caller's own error surface

    candidates = []
    for username in sorted(os.listdir(_DATASETS_PREFIX)):
        if username == stale_username:
            continue  # already confirmed not to exist, above
        # Built with an explicit "/" rather than os.path.join -- this whole module
        # only ever runs on Kaggle (Linux), and the rest of this function already
        # treats _DATASETS_PREFIX/path as forward-slash strings (the startswith
        # check above, the rest = path[len(prefix):] slice) -- os.path.join would
        # silently mix separator styles on a non-Linux dev machine.
        candidate = f"{_DATASETS_PREFIX}{username}/{remainder}"
        if os.path.isdir(candidate):
            candidates.append(candidate)

    if len(candidates) == 1:
        log.warning(
            "resolve_kaggle_dataset_path: %r doesn't exist, but found exactly one match under "
            "a different username -- using %r instead. The username segment of "
            "/kaggle/input/datasets/<username>/... has been observed to change between Kaggle "
            "sessions for the same attached dataset; consider updating the config to this path "
            "if it stays stable.",
            path, candidates[0],
        )
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"{path!r} doesn't exist, and more than one username-substituted candidate does: "
            f"{candidates} -- can't guess which is correct. Fix the config path explicitly."
        )
    return path  # no substitute found either -- let the caller's existing "not a directory" error surface, unchanged
