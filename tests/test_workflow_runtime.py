from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CHECKOUT_REF = (
    "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
)
SETUP_PYTHON_REF = (
    "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405"
)


def test_workflows_are_valid_yaml_and_use_node24_pinned_actions() -> None:
    paths = sorted(WORKFLOWS.glob("*.yml"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert yaml.safe_load(text) is not None
        for line in text.splitlines():
            stripped = line.strip()
            normalized = stripped.removeprefix("- ").strip()
            if "uses: actions/checkout@" in normalized:
                assert normalized == f"uses: {CHECKOUT_REF}"
            if "uses: actions/setup-python@" in normalized:
                assert normalized == f"uses: {SETUP_PYTHON_REF}"

    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "actions/checkout@v4" not in combined
    assert "actions/setup-python@v5" not in combined
    assert "11d5960a326750d5838078e36cf38b85af677262" not in combined
    assert "a26af69be951a213d495a4c3e4e4022e16d87065" not in combined
