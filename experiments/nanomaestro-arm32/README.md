# NanoMaestro Light: isolated ARM32 feasibility probe

This experiment does not replace any Y2Synth instrument or publish a new Pages UI.
FLOW remains paused. No weights or generated audio are committed.

## What the workflow checks

1. Use the pinned Hugging Face revision and verified SHA-256; download the matching original
   Light `model_int8.onnx` + `vocab.json`, recording sizes and SHA-256.
2. Check the ONNX graph, operators, input shapes and quantized initializer types.
3. Run 256 stateful steps on x86 with Debian's CPU ONNX Runtime, creating a fixed-input
   teacher-forced fixture. Record logits and final hidden/cell state.
4. Cross compile the C++ probe for Linux ARMv7 hard-float, Cortex-A7, NEON/VFPv4,
   linking Debian trixie's ARM32 ONNX Runtime package. Execute through QEMU's
   `cortex-a7` CPU model. The x86 reference uses Debian's matching runtime version.
5. Compare all 256 logits vectors, record final state errors, then independently
   sample another 512 event tokens with recurrent state preserved.

Reference acceptance: finite outputs throughout; maximum softmax probability
difference <= 0.02 and >= 90% matching argmax decisions on the fixed trajectory.
These are declared cross-backend quantized-model tolerances, not exact numerical
equivalence. Raw observed errors and match counts are always reported.

The model revision, graph report, Debian package versions, ELF/ABI metadata, sampled
token text and logs are uploaded as a 14-day Actions artifact. Intermediate
weights, fixture binaries, runtime libraries and executables stay on the runner.

## What this does NOT establish

- QEMU time is not Y2 inference time, even when `-cpu cortex-a7` is used.
- Emulated process max RSS is not measured hardware memory consumption.
- This glibc/Linux binary does not run directly under stock Android 4.4/Bionic.
- Dynamic-quantization support on ARM32 must pass the actual session/run check.
- We do not test audio scheduling, tone quality, token grammar recovery, battery,
  OS/driver integration or a real Innioasis Y2.
- Model size plus recurrent state bytes is not the complete runtime RAM budget.

The 512 sampled tokens are a neural execution smoke test, not a scored musical
performance or a ready player. A real-device gate must measure inference rate,
worst-case buffer underruns and peak RSS alongside the intended audio engine.

## Run

Push a change in this directory, or run the **NanoMaestro Light ARM32 feasibility**
workflow manually. The standalone probe builds remotely in GitHub Actions; no local checkout
or installation is required.

Sources:
- [NanoMaestro model](https://huggingface.co/utkucoban/NanoMaestro-Realtime)
- [Light files](https://huggingface.co/utkucoban/NanoMaestro-Realtime/tree/main/NanoMaestro-Light)
- [Author's CPU example](https://huggingface.co/utkucoban/NanoMaestro-Realtime/blob/main/Example%20inference%20code/local-demo-CPU.py)
- [ONNX Runtime build](https://onnxruntime.ai/docs/build/inferencing.html)

## Runtime source availability

On 21 November 2026 the official ONNX Runtime Git repository and v1.23.2
codeload archive returned errors from the GitHub runner. The first x86 reference
passed using PyPI 1.23.2, but source compilation could not start. The current
workflow uses signed Debian trixie packages for both x86 and armhf instead.
This tests the distribution's full runtime, not our initially proposed reduced
source build. Exact installed versions are included in the artifact.
