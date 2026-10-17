# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Single-file access to the g-xTB v2.0.1 parameter archive.

All g-xTB v2.0.1 tables live in one archive, ``params/gxtb_v2.npz``. Four
namespaces share it; ``cov_radii`` and ``max_z`` occur in more than one, so
three of them carry a prefix. :func:`load_tables` returns one namespace with
that prefix stripped, which is what each typed accessor module wants.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "params", "gxtb_v2.npz")

#: namespace -> the name prefixes belonging to it. The g-xTB tables are
#: already namespaced by ``pa_``/``ps_``/``pg_`` and carry no extra prefix.
NAMESPACES: Dict[str, tuple[str, ...]] = {
    "gxtb": ("pa_", "ps_", "pg_"),
    "qvszp": ("qvszp_",),
    "eeqbc": ("eeqbc_",),
    "mctc": ("mctc_",),
}


def load_tables(
    namespace: str, path: str | os.PathLike[str] | None = None
) -> Dict[str, np.ndarray]:
    """Return one namespace's tables, keyed by their own names."""

    try:
        prefixes = NAMESPACES[namespace]
    except KeyError:
        raise KeyError(
            "unknown namespace %r; expected one of %s"
            % (namespace, ", ".join(sorted(NAMESPACES)))
        ) from None
    # Only a namespace given a prefix of its own has one to strip.
    strip = len(prefixes[0]) if namespace != "gxtb" else 0
    with np.load(path or ARCHIVE_PATH, allow_pickle=False) as data:
        tables = {
            name[strip:]: data[name].copy()
            for name in data.files
            if name.startswith(prefixes)
        }
    if not tables:
        raise ValueError("no %s tables in %s" % (namespace, path or ARCHIVE_PATH))
    return tables


def provenance(path: str | os.PathLike[str] | None = None) -> str:
    """Return the archive's provenance note."""

    with np.load(path or ARCHIVE_PATH, allow_pickle=False) as data:
        return str(data["__provenance__"])
