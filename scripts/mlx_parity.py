"""Run the native MLX cold model against an exported reference parity suite."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    load_asset_bundle,
)
from streamner_commit.mlx.decoder import DecodedSpan, SpanDecoder, sigmoid
from streamner_commit.mlx.model import MLXColdModel
from streamner_commit.mlx.precision import require_mlx_full_precision
from streamner_commit.mlx.qwen_adapter import numerical_diagnostics
from streamner_commit.reference.parity import (
    ARRAYS_FILENAME,
    METADATA_FILENAME,
    REQUIRED_ARRAY_NAMES,
    validate_arrays,
)

DISCRETE_ARRAYS = (
    "input_ids",
    "attention_mask",
    "label_attention_mask",
    "words_mask",
    "text_lengths",
    "span_idx",
    "span_mask",
    "label_token_positions",
    "separator_token_positions",
)
METADATA_FIELDS = (
    "labels",
    "word_tokens",
    "word_char_starts",
    "word_char_ends",
    "serialized_prompt",
    "serialized_prompt_words",
    "serialized_input_words",
    "prompt_word_length",
    "tokenizer_tokens",
)
THRESHOLDS = (0.3, 0.5, 0.7)
MINIMUM_COSINE = 0.99999
TOP_LABEL_TIE_MARGIN = 0.1
TOP_LABEL_NEGLIGIBLE_PROBABILITY = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("artifacts/reference") / REFERENCE_REVISION
    parser.add_argument("--asset-root", type=Path, default=default_root)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return dict(value)


def _load_case(case_dir: Path) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    metadata = _json_object(case_dir / METADATA_FILENAME, name="parity metadata")
    with np.load(case_dir / ARRAYS_FILENAME, allow_pickle=False) as archive:
        arrays = validate_arrays(
            {name: archive[name] for name in archive.files},
            required=REQUIRED_ARRAY_NAMES,
        )
    return arrays, metadata


def _span_signature(spans: Sequence[DecodedSpan]) -> list[list[Any]]:
    return [[span.start_word, span.end_word, span.label] for span in spans]


def _score_delta(
    reference: Sequence[DecodedSpan],
    candidate: Sequence[DecodedSpan],
) -> float | None:
    if _span_signature(reference) != _span_signature(candidate):
        return None
    return max(
        (abs(left.score - right.score) for left, right in zip(reference, candidate, strict=True)),
        default=0.0,
    )


def _diagnostic(reference: np.ndarray[Any, Any], candidate: Any) -> dict[str, Any]:
    return numerical_diagnostics(reference, np.asarray(candidate)).to_dict()


def _preprocessing_metadata(prepared: Any) -> dict[str, Any]:
    return {
        "labels": list(prepared.labels),
        "word_tokens": list(prepared.word_tokens),
        "word_char_starts": list(prepared.word_char_starts),
        "word_char_ends": list(prepared.word_char_ends),
        "serialized_prompt": prepared.serialized_prompt,
        "serialized_prompt_words": list(prepared.serialized_prompt_words),
        "serialized_input_words": list(prepared.serialized_input_words),
        "prompt_word_length": prepared.prompt_word_length,
        "tokenizer_tokens": list(prepared.tokenizer_tokens),
    }


def main() -> None:
    require_mlx_full_precision()
    args = parse_args()
    asset_root = args.asset_root.resolve()
    suite_dir = (args.suite_dir or asset_root / "parity_suite").resolve()
    output_path = (args.output or asset_root / "mlx_cold_parity_report.json").resolve()
    manifest = _json_object(suite_dir / "suite_manifest.json", name="suite manifest")
    if manifest.get("model_id") != REFERENCE_MODEL_ID:
        raise ValueError("suite model ID does not match the locked reference")
    if manifest.get("model_revision_sha") != REFERENCE_REVISION:
        raise ValueError("suite revision does not match the locked reference")
    case_records = manifest.get("cases")
    if not isinstance(case_records, list) or not case_records:
        raise ValueError("suite manifest must contain case records")

    bundle = load_asset_bundle(
        asset_root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    model = MLXColdModel.from_asset_bundle(bundle)
    decoder = SpanDecoder()
    results: list[dict[str, Any]] = []
    all_pass = True

    for raw_record in case_records:
        if not isinstance(raw_record, Mapping):
            raise TypeError("suite case records must be objects")
        case_id = raw_record.get("id")
        directory = raw_record.get("directory")
        if not isinstance(case_id, str) or not isinstance(directory, str):
            raise ValueError("suite case records require string IDs and directories")
        arrays, metadata = _load_case(suite_dir / directory)
        text = metadata.get("text")
        labels = metadata.get("labels")
        if not isinstance(text, str) or not isinstance(labels, list):
            raise ValueError(f"case {case_id} metadata lacks text or labels")

        forward = model.forward(text, labels)
        prepared = forward.preprocessing
        prepared_arrays = dict(prepared.as_model_inputs())
        prepared_arrays.update(
            {
                "label_token_positions": prepared.label_token_positions,
                "separator_token_positions": prepared.separator_token_positions,
            }
        )
        discrete = {
            name: bool(np.array_equal(prepared_arrays[name], arrays[name]))
            for name in DISCRETE_ARRAYS
        }
        actual_metadata = _preprocessing_metadata(prepared)
        metadata_equal = {
            name: actual_metadata[name] == metadata.get(name) for name in METADATA_FIELDS
        }

        numerical = {
            "qwen_final_hidden_states": _diagnostic(
                arrays["qwen_final_hidden_states"], forward.qwen_hidden_states
            ),
            "prompt_input_hidden_states": _diagnostic(
                arrays["prompt_input_hidden_states"], forward.labels.prompt.hidden_states
            ),
            "contextualized_prompt_hidden_states": _diagnostic(
                arrays["contextualized_prompt_hidden_states"],
                forward.labels.contextualized_prompt,
            ),
            "label_representations_pre_projection": _diagnostic(
                arrays["label_representations_pre_projection"],
                forward.labels.label_representations,
            ),
            "pooled_word_states": _diagnostic(
                arrays["pooled_word_states"], forward.pooled_word_states
            ),
            "marker_v2_span_representations": _diagnostic(
                arrays["marker_v2_span_representations"], forward.span_states
            ),
            "label_representations_post_projection": _diagnostic(
                arrays["label_representations_post_projection"],
                forward.projected_label_states,
            ),
            "raw_logits": _diagnostic(arrays["raw_logits"], forward.logits),
        }

        reference_logits = arrays["raw_logits"]
        candidate_logits = np.asarray(forward.logits)
        valid = arrays["span_mask"].reshape(-1).astype(bool)
        reference_top = reference_logits.reshape(-1, reference_logits.shape[-1])[valid].argmax(
            axis=-1
        )
        candidate_top = candidate_logits.reshape(-1, candidate_logits.shape[-1])[valid].argmax(
            axis=-1
        )
        top_matches = int(np.sum(reference_top == candidate_top))
        top_total = int(reference_top.size)
        valid_positions = np.flatnonzero(valid)
        mismatch_positions = valid_positions[reference_top != candidate_top]
        flat_reference = reference_logits.reshape(-1, reference_logits.shape[-1])
        flat_candidate = candidate_logits.reshape(-1, candidate_logits.shape[-1])
        flat_boundaries = arrays["span_idx"].reshape(-1, 2)
        top_label_mismatches: list[dict[str, Any]] = []
        material_top_label_mismatches = 0
        for position in mismatch_positions:
            reference_order = np.argsort(-flat_reference[position], kind="stable")
            candidate_order = np.argsort(-flat_candidate[position], kind="stable")
            reference_margin = float(
                flat_reference[position, reference_order[0]]
                - flat_reference[position, reference_order[1]]
            )
            reference_probability = float(sigmoid(flat_reference[position, reference_order[0]]))
            negligible_near_tie = (
                reference_margin <= TOP_LABEL_TIE_MARGIN
                and reference_probability <= TOP_LABEL_NEGLIGIBLE_PROBABILITY
            )
            material_top_label_mismatches += int(not negligible_near_tie)
            boundary = flat_boundaries[position]
            top_label_mismatches.append(
                {
                    "candidate_position": int(position),
                    "boundary": [int(boundary[0]), int(boundary[1])],
                    "reference_label": labels[int(reference_order[0])],
                    "mlx_label": labels[int(candidate_order[0])],
                    "reference_logit_margin": reference_margin,
                    "reference_top_probability": reference_probability,
                    "negligible_near_tie": negligible_near_tie,
                }
            )

        decoder_results: dict[str, Any] = {}
        canonical_match = False
        canonical_score_delta: float | None = None
        for threshold in THRESHOLDS:
            reference_spans = decoder.decode(
                reference_logits,
                arrays["span_idx"],
                arrays["span_mask"],
                labels,
                threshold=threshold,
            )[0]
            candidate_spans = decoder.decode(
                candidate_logits,
                prepared.span_idx,
                prepared.span_mask,
                prepared.labels,
                threshold=threshold,
            )[0]
            signature_match = _span_signature(reference_spans) == _span_signature(candidate_spans)
            score_delta = _score_delta(reference_spans, candidate_spans)
            decoder_results[str(threshold)] = {
                "reference": _span_signature(reference_spans),
                "mlx": _span_signature(candidate_spans),
                "boundary_label_match": signature_match,
                "max_probability_error": score_delta,
            }
            if threshold == 0.5:
                canonical_match = signature_match
                canonical_score_delta = score_delta

        numerical_pass = all(
            item["shape_equal"]
            and item["cosine_similarity"] is not None
            and item["cosine_similarity"] >= MINIMUM_COSINE
            for item in numerical.values()
        )
        case_pass = (
            all(discrete.values())
            and all(metadata_equal.values())
            and numerical_pass
            and material_top_label_mismatches == 0
            and canonical_match
        )
        all_pass &= case_pass
        results.append(
            {
                "id": case_id,
                "pass": case_pass,
                "discrete_exact": discrete,
                "metadata_exact": metadata_equal,
                "candidate_count": top_total,
                "top_label_matches": top_matches,
                "top_label_agreement": top_matches / top_total if top_total else 1.0,
                "top_label_mismatches": top_label_mismatches,
                "material_top_label_mismatches": material_top_label_mismatches,
                "numerical": numerical,
                "decoder": decoder_results,
                "canonical_max_probability_error": canonical_score_delta,
            }
        )
        print(
            f"{case_id}: {'PASS' if case_pass else 'FAIL'}; "
            f"top labels {top_matches}/{top_total}; public@0.5={canonical_match}"
        )

    report = {
        "schema_version": 1,
        "kind": "mlx_cold_parity_report",
        "model_id": bundle.model_id,
        "model_revision_sha": bundle.revision,
        "minimum_component_cosine": MINIMUM_COSINE,
        "top_label_near_tie_rule": {
            "maximum_reference_logit_margin": TOP_LABEL_TIE_MARGIN,
            "maximum_reference_top_probability": TOP_LABEL_NEGLIGIBLE_PROBABILITY,
        },
        "canonical_threshold": 0.5,
        "case_count": len(results),
        "pass": all_pass,
        "cases": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    if not all_pass:
        raise SystemExit("MLX cold parity gate failed")


if __name__ == "__main__":
    main()
