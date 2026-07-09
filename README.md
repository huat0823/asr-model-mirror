# ASR Model Mirror

## Quickstart (one command)

```bash
curl -fsSL --retry 10 --retry-delay 20 --retry-all-errors \
  https://raw.githubusercontent.com/huat0823/asr-model-mirror/main/setup_asr.py -o setup_asr.py \
  && python3 setup_asr.py            # add --large for the 243MB model
```

`setup_asr.py` handles everything: 429/503 backoff, resume on connection cuts (HTTP Range),
SHA256 verification, part reassembly, `pip install sherpa-onnx numpy`, and a recognition
smoke test. Idempotent — just re-run it after any failure.

Mirror of Chinese ASR models for restricted-network environments (only `raw.githubusercontent.com` + PyPI reachable). All files are plain git blobs — **no LFS** — so they are served directly by `raw.githubusercontent.com`.

## Contents

| Dir | Model | Size | Use |
|---|---|---|---|
| `paraformer-zh-small/` | sherpa-onnx-paraformer-zh-small-2024-03-09 (int8) | 82 MB | Fast, good zh + code-switched en |
| `paraformer-zh-large/` | sherpa-onnx-paraformer-zh-2023-09-14 (int8), split into `.part-aa/ab/ac` | 243 MB | Best accuracy zh+en |
| `vad/` | silero_vad.onnx | 0.6 MB | VAD for long audio |
| `test/2-zh-en.wav` | test clip | — | sanity check |

Reassemble the large model: `cat model.int8.onnx.part-* > model.int8.onnx`

Verify with `SHA256SUMS` (includes the hash of the assembled large model).

## Runtime

```
pip install sherpa-onnx
```

## Source & license

Models exported by the [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project (release tag `asr-models`); Paraformer originates from Alibaba FunASR, Silero VAD from snakers4/silero-vad. Redistributed unmodified for mirror purposes under their original licenses (Apache-2.0 / MIT). Not affiliated with the upstream projects.
