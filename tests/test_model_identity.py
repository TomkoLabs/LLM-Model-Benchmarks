from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from harness.model_identity import (
    ModelMetadataError,
    extract_runtime_metadata,
    load_registry,
    resolve_public_identity,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def registry(*, include_second: bool = False) -> dict:
    models = {
        DIGEST_A: {
            "canonical_name": "Example Coder",
            "source_model": "Example/Example-Coder-32B-v2",
            "source_url": "https://example.invalid/Example-Coder-32B-v2",
            "version": "v2",
            "parameter_variant": "32B",
            "architecture": "examplemoe",
            "quantization": "Q8_0",
            "source_filename": "example-coder-v2-q8_0.gguf",
            "aliases": ["agent-main:latest"],
        }
    }
    if include_second:
        models[DIGEST_B] = {
            "canonical_name": "Example Coder",
            "source_model": "Example/Example-Coder-32B-v2",
            "source_url": "https://example.invalid/Example-Coder-32B-v2",
            "version": "v2",
            "parameter_variant": "32B",
            "architecture": "examplemoe",
            "quantization": "Q4_K_M",
            "source_filename": "example-coder-v2-q4_k_m.gguf",
            "aliases": ["agent-main:latest"],
        }
    return {"schema_version": 1, "models": models}


class ModelIdentityTests(unittest.TestCase):
    def _model(self, digest: str = DIGEST_A, quantization: str = "Q8_0") -> dict:
        return {
            "runtime_model": "agent-main:latest",
            "runtime_digest": digest,
            "quantization": quantization,
            "context_length": 131072,
            "reasoning_effort": "medium",
        }

    def test_alias_resolves_by_digest_to_full_canonical_identity(self) -> None:
        identity = resolve_public_identity(self._model(), registry=registry())
        self.assertEqual(identity["status"], "COMPLETE")
        self.assertEqual(identity["canonical_name"], "Example Coder")
        self.assertIn("Example Coder v2", identity["display_name"])
        self.assertIn("32B (examplemoe)", identity["display_name"])
        self.assertIn("Q8_0", identity["display_name"])
        self.assertIn("131072-token context", identity["display_name"])
        self.assertIn(DIGEST_A, identity["display_name"])
        self.assertEqual(identity["runtime_alias"], "agent-main:latest")

    def test_same_alias_repointed_to_unknown_digest_is_not_inferred(self) -> None:
        identity = resolve_public_identity(
            self._model(DIGEST_B), registry=registry()
        )
        self.assertEqual(identity["status"], "METADATA_INCOMPLETE")
        self.assertIsNone(identity["canonical_name"])
        self.assertNotIn("Example Coder", identity["display_name"])
        self.assertIn(
            "canonical checkpoint name is unavailable", identity["issues"]
        )

    def test_same_checkpoint_quantizations_remain_distinct(self) -> None:
        full_registry = registry(include_second=True)
        q8 = resolve_public_identity(self._model(), registry=full_registry)
        q4 = resolve_public_identity(
            self._model(DIGEST_B, "Q4_K_M"), registry=full_registry
        )
        self.assertEqual(q8["source_model"], q4["source_model"])
        self.assertNotEqual(q8["quantization"], q4["quantization"])
        self.assertNotEqual(q8["immutable_digest"], q4["immutable_digest"])
        self.assertNotEqual(q8["display_name"], q4["display_name"])

    def test_registry_lookup_is_digest_primary(self) -> None:
        identity = resolve_public_identity(
            {**self._model(), "runtime_model": "an-unregistered-alias:latest"},
            registry=registry(),
        )
        self.assertEqual(identity["status"], "COMPLETE")
        self.assertFalse(identity["alias_registered"])
        self.assertEqual(identity["canonical_name"], "Example Coder")

    def test_missing_metadata_never_fabricates_name_from_alias(self) -> None:
        identity = resolve_public_identity(
            {
                "runtime_model": "looks-like-a-famous-model-70b:latest",
                "runtime_digest": DIGEST_B,
                "quantization": "Q8_0",
                "context_length": 4096,
            },
            registry=registry(),
        )
        self.assertIsNone(identity["canonical_name"])
        self.assertEqual(identity["status"], "METADATA_INCOMPLETE")
        self.assertNotIn("famous", identity["display_name"])

    def test_absolute_gguf_path_is_reduced_to_safe_basename(self) -> None:
        metadata = extract_runtime_metadata(
            {
                "details": {
                    "quantization_level": "Q8_0",
                    "parent_model": "/home/private/models/parent.gguf",
                }
            },
            {
                "modelfile": 'FROM "/home/private/models/example coder.gguf"\n',
                "model_info": {"general.basename": "Example Coder"},
            },
        )
        self.assertEqual(metadata["source_filename"], "example coder.gguf")
        self.assertNotIn("source_model", metadata)
        self.assertNotIn("/home/private", str(metadata))

    def test_runtime_metadata_extracts_exact_num_ctx_and_quantization(self) -> None:
        metadata = extract_runtime_metadata(
            {
                "details": {
                    "family": "examplemoe",
                    "parameter_size": "35.7B",
                    "quantization_level": "Q6_K_L",
                    "context_length": 262144,
                }
            },
            {
                "details": {
                    "family": "examplemoe",
                    "parameter_size": "35.7B",
                    "quantization_level": "Q6_K_L",
                },
                "parameters": "temperature 0.7\nnum_ctx 65536\n",
                "modelfile": (
                    "FROM /private/models/example.gguf\n"
                    "PARAMETER num_ctx 65536\n"
                ),
                "model_info": {
                    "general.architecture": "examplemoe",
                    "examplemoe.context_length": 262144,
                },
            },
        )
        self.assertEqual(metadata["quantization"], "Q6_K_L")
        self.assertEqual(metadata["effective_context_length"], 65536)
        self.assertEqual(metadata["native_context_length"], 262144)
        self.assertEqual(
            metadata["effective_context_sources"],
            [
                "/api/show parameters num_ctx",
                "/api/show Modelfile PARAMETER num_ctx",
            ],
        )

    def test_conflicting_num_ctx_is_not_treated_as_effective(self) -> None:
        metadata = extract_runtime_metadata(
            {"details": {"family": "dense"}},
            {
                "parameters": "num_ctx 4096\n",
                "modelfile": "PARAMETER num_ctx 8192\n",
            },
        )
        self.assertNotIn("effective_context_length", metadata)
        self.assertEqual(
            metadata["metadata_discrepancies"],
            [
                "conflicting num_ctx values in /api/show parameters and Modelfile"
            ],
        )

    def test_complete_runtime_metadata_can_supply_public_identity(self) -> None:
        identity = resolve_public_identity(
            self._model(DIGEST_B),
            registry=registry(),
            runtime_metadata={
                "canonical_name": "Runtime Coder",
                "source_model": "Example/Runtime-Coder-v4",
                "version": "v4",
                "parameter_variant": "32B",
                "architecture": "examplemoe",
                "quantization": "Q8_0",
                "source_filename": "runtime-coder-q8.gguf",
            },
        )
        self.assertEqual(identity["status"], "COMPLETE")
        self.assertFalse(identity["registry_match"])
        self.assertEqual(identity["canonical_name"], "Runtime Coder")
        self.assertIn(DIGEST_B, identity["display_name"])

    def test_runtime_ready_identity_retains_partial_metadata(self) -> None:
        identity = resolve_public_identity(
            self._model(DIGEST_B),
            registry=registry(),
            runtime_metadata={
                "parameter_variant": "32B",
                "architecture": "examplemoe",
                "quantization": "Q8_0",
            },
        )
        self.assertEqual(identity["status"], "METADATA_INCOMPLETE")
        self.assertEqual(identity["parameter_variant"], "32B")
        self.assertEqual(identity["architecture"], "examplemoe")
        self.assertIn(
            "canonical checkpoint name is unavailable", identity["issues"]
        )

    def test_historical_enrichment_does_not_mutate_original_result(self) -> None:
        historical = {
            "model": self._model(),
            "provenance": {"model_config": {"context_length": 131072}},
        }
        before = copy.deepcopy(historical)
        identity = resolve_public_identity(
            historical["model"], historical["provenance"], registry=registry()
        )
        self.assertEqual(identity["status"], "COMPLETE")
        self.assertEqual(historical, before)

    def test_runtime_metadata_mismatch_fails_public_completeness(self) -> None:
        identity = resolve_public_identity(
            self._model(),
            registry=registry(),
            runtime_metadata={"quantization": "Q4_K_M"},
        )
        self.assertEqual(identity["status"], "METADATA_INCOMPLETE")
        self.assertIn(
            "registered quantization does not match run metadata",
            identity["issues"],
        )

    def test_registry_rejects_missing_fields_and_private_paths(self) -> None:
        for mutation in ("missing-version", "private-path"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                value = registry()
                entry = value["models"][DIGEST_A]
                if mutation == "missing-version":
                    del entry["version"]
                else:
                    entry["source_model"] = "/home/operator/private.gguf"
                path = Path(temporary) / "model-metadata.yaml"
                path.write_text(
                    yaml.safe_dump(value, sort_keys=True), encoding="utf-8"
                )
                with self.assertRaises(ModelMetadataError):
                    load_registry(path)


if __name__ == "__main__":
    unittest.main()
