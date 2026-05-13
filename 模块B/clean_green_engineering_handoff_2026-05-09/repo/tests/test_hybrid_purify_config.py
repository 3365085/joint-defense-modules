from pathlib import Path

import yaml


def test_hybrid_purify_config_has_required_sections():
    # Resolve relative to the repo root (parents[1] = repo/) so the test
    # runs regardless of the pytest cwd. The handoff-package-relative
    # layout is ``repo/tests/test_*.py`` + ``repo/configs/*.yaml``.
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "hybrid_purify_detox.yaml"
    assert cfg_path.exists()
    data = yaml.safe_load(cfg_path.read_text())
    cfg = data["hybrid_purify_detox"]
    assert cfg["max_allowed_external_asr"] <= 0.10
    attacks = cfg["attacks"]
    names = {a["name"] for a in attacks}
    assert {"badnet_oga", "blend_oga", "wanet_oga", "badnet_oda", "semantic_green_cleanlabel"}.issubset(names)
    for a in attacks:
        if a["goal"] == "oga":
            assert a.get("poison_negative") is True
            assert a.get("poison_positive") is False
        if a["goal"] == "oda":
            assert a.get("poison_positive") is True
            assert a.get("poison_negative") is False
