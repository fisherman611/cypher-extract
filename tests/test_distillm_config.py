from pathlib import Path

import yaml

ADAPTIVE_CONFIGS = sorted(Path("configs").glob("*/distillm_adaptive_*.yaml"))


def test_every_family_has_both_adaptive_distillm_configs() -> None:
    assert len(ADAPTIVE_CONFIGS) == 24


def test_distillm_loss_eps_matches_the_dev_loss_scale() -> None:
    # CypherKD's loss_eps of 0.1 is meant for its larger Cypher-only dev loss. The multitask dev
    # loss here is about 0.005-0.03 per epoch (measured on the qwen3 LoRA students), so 0.1 would
    # never raise the adaptive threshold and the run would silently be plain sfkl/srkl. loss_eps
    # must stay below the epoch-1 dev loss. The same value is used for every family.
    for path in ADAPTIVE_CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["init_threshold"] == 0.0, path
        assert 0.0 < config["loss_eps"] <= 0.01, path
