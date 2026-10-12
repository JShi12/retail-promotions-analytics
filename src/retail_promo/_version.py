"""Code fingerprint: a short hash of the package's source files.

Databricks Git folders do not expose `.git`, so notebooks print this instead of a commit id.
`CODE_FINGERPRINT` in `retail_promo/__init__.py` is computed when the package is first imported;
if the files on disk change afterwards (a Git pull in a live session), `check_fresh()` notices
that the session is still running the old code.

    uv run python -c "import retail_promo; print(retail_promo.CODE_FINGERPRINT)"
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        h.update(path.relative_to(PACKAGE_DIR).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:10]


def check_fresh() -> str:
    """Print the code fingerprint; fail if the session imported an older version of the code."""
    import retail_promo

    on_disk = fingerprint()
    print(f"code fingerprint: {retail_promo.CODE_FINGERPRINT}")
    if on_disk != retail_promo.CODE_FINGERPRINT:
        raise RuntimeError(
            f"The code changed on disk ({on_disk}) after this Python session imported it "
            f"({retail_promo.CODE_FINGERPRINT}), e.g. after a Git pull. Detach and reattach the "
            "notebook (or run dbutils.library.restartPython()), then Run all."
        )
    return on_disk
