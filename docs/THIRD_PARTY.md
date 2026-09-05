# Data and dependencies

- **Model:** [knowledgator/gliner-stream-pii-v1.0](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0),
  revision `e871777dc4b3b688747a0433fff8d94a36fcc7b0`, published under Apache-2.0.
  Its Qwen3 backbone and checkpoint-specific tensors are loaded from separately obtained assets.
- **Reference implementation:** [GLiNER](https://github.com/urchade/GLiNER) 0.2.28,
  Apache-2.0, used in a separate PyTorch environment.
- **Runtime:** [MLX](https://github.com/ml-explore/mlx) 0.32.2 and
  [MLX-LM](https://github.com/ml-explore/mlx-lm) 0.31.3, MIT. The code imports
  MLX-LM's Qwen3 and cache implementations.
- **Tokenizer:** Transformers 5.12.1. Full dependency pins are in `../uv.lock`.

No dependency source tree is vendored. Model weights and tokenizer exports are not
redistributed. Dependency and model licenses remain separate from this project's code.

## Dataset

[PIIMB](https://huggingface.co/datasets/piimb/pii-masking-benchmark) is **CC BY-NC 4.0**
and downloaded separately. Dataset text and annotations are not redistributed.
Its underlying source terms also apply: Ai4Privacy OpenPII 1M (Ai4Privacy/Ai Suisse SA,
CC BY 4.0), Gretel PII Masking EN v1 (Gretel.ai, Apache 2.0), Nemotron-PII (NVIDIA,
CC BY 4.0), and Privy (Benjamin Kilimnik, MIT).

The source revision and task configuration are pinned in `../configs/piimb.json`.
Selection manifests contain IDs, split metadata, and checksums, not corpus text.
The small fixtures under `data/fixtures/` are fictional debugging examples.

## Other references

[gliner2-mlx](https://github.com/Andrew-Chen-Wang/gliner2-mlx) and
[OpenMed](https://github.com/maziyarpanahi/openmed) were consulted as technical references;
their source was not copied or adapted. Incremental evaluation was informed by
[Madureira, Kahardipraja, and Schlangen (2023)](https://aclanthology.org/2023.sigdial-1.14/).

No project-level software license has been selected. Public visibility is not itself
an open-source license. This tool is not a production privacy or compliance guarantee.
