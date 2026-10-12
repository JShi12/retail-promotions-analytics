import pytest

import retail_promo
from retail_promo import _version


def test_fingerprint_is_stable_and_matches_import():
    assert _version.fingerprint() == _version.fingerprint()
    assert retail_promo.CODE_FINGERPRINT == _version.fingerprint()
    assert _version.check_fresh() == retail_promo.CODE_FINGERPRINT


def test_check_fresh_detects_a_stale_session(monkeypatch):
    # Simulates a Git pull after import: the imported fingerprint no longer matches the files.
    monkeypatch.setattr(retail_promo, "CODE_FINGERPRINT", "0000000000")
    with pytest.raises(RuntimeError, match="Detach and reattach"):
        _version.check_fresh()
