"""Public aggregate export checks; no private corpus, model or replay required."""

import json
import math
import re
from pathlib import Path

from streamner_commit.metrics.accuracy import MaskingMetrics, StrictNERMetrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/overnight/2026-09-06"
RATE_KEYS = (
    "masking_precision",
    "masking_recall",
    "masking_f1",
    "masking_f2",
    "fpr",
    "strict_precision",
    "strict_recall",
    "strict_f1",
)


def data():
    return json.loads((RESULTS / "metrics.json").read_text())


def test_overnight_public_rates_and_complete_coverage():
    p = data()
    assert p["status"] == "complete"
    assert p["planned_policy_pairs"] == p["complete_policy_pairs"] == 14400
    assert p["sample"]["examples"] == 1200 and p["sample"]["parents"] == 1160
    assert len(p["coverage"]) == 4 and all(r["common"] == 1200 for r in p["coverage"])
    assert len(p["metrics"]) == 160 and len(p["task_averages"]) == 32
    assert not p["provenance"]["final_reproducibility_gate"]
    seen = set()
    for r in p["metrics"]:
        key = r["chunk_words"], r["control"], r["phase"], r["task"]
        assert key not in seen
        seen.add(key)
        c = r["counts"]
        strict = StrictNERMetrics(c["tp"], c["fp"], c["fn"])
        masking = MaskingMetrics(
            c["mask_tp"], c["mask_predicted"], c["mask_gold"], c["mask_non_pii"]
        )
        expected = (
            masking.precision,
            masking.recall,
            masking.f1,
            masking.f2,
            masking.fpr,
            strict.precision,
            strict.recall,
            strict.f1,
        )
        for name, value in zip(RATE_KEYS, expected, strict=True):
            assert math.isclose(r[name], value, abs_tol=1e-12)


def test_task_macro_and_pooled_micro_are_not_confused():
    p = data()
    for average in p["task_averages"]:
        common = (average["chunk_words"], average["control"], average["phase"])
        rows = [r for r in p["metrics"] if (r["chunk_words"], r["control"], r["phase"]) == common]
        tasks = [r for r in rows if r["task"] != "overall"]
        pooled = next(r for r in rows if r["task"] == "overall")
        assert len(tasks) == 4 and {r["task"] for r in tasks} == set(p["sample"]["per_task"])
        assert all(r["examples"] == 300 for r in tasks)
        for k in RATE_KEYS:
            assert math.isclose(average[k], sum(r[k] for r in tasks) / 4, abs_tol=1e-12)
        for k in pooled["counts"]:
            assert pooled["counts"][k] == sum(r["counts"][k] for r in tasks)


def test_readme_matches_masking_not_strict_columns_and_preserves_caveats():
    p = data()
    hf = json.loads((RESULTS / "hf_reference.json").read_text())
    readme = (ROOT / "README.md").read_text()
    selected = [
        r
        for r in p["task_averages"]
        if r["chunk_words"] == 1
        and (
            r["control"] == "buffer_2"
            or (r["control"], r["phase"])
            in (("cold_full", "full_text"), ("ema_published", "online"))
        )
    ]
    for r in (*selected, hf["metrics"][0]):
        values = [r[k] for k in (*RATE_KEYS[:5], "strict_f1")]
        rendered = " | ".join("—" if v is None else f"{v * 100:.2f}%" for v in values)
        assert rendered in readme
    assert "not a matched benchmark reproduction" in readme
    assert "not live detection" in readme and "not pooled scores" in readme
    assert hf["threshold"] == 0.5 and hf["dtype"] == "bfloat16"
    assert all(spec["parameters"]["threshold"] == 0.95 for spec in p["controls"].values())
    assert hf["model_card_revision"] == p["model"]["revision"]
    assert hf["dataset_revision"] == p["dataset"]["revision"]


def test_public_export_has_no_case_identifiers_or_local_paths():
    forbidden_keys = {
        "example",
        "example_id",
        "parent",
        "parent_id",
        "selected",
        "reserved",
        "text",
        "span_text",
        "hostname",
        "checkpoint_compatibility",
        "traceback",
    }

    def check(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str):
            assert not any(s in value for s in ("/Users/", "/home/", ".private/"))
            assert not re.search(r"\b10\.\d+\.\d+\.\d+\b", value)

    for name in ("metrics.json", "hf_reference.json"):
        check(json.loads((RESULTS / name).read_text()))
