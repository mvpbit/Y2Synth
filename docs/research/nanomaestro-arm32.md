# NanoMaestro Light: ARM32 feasibility evidence — 21 November 2026

> Historical ARM32 experiment. Later on the same date, the user authorized
> browser integration; NanoMaestro FLOW is now published on Pages.
> See [DEVELOPMENT](../DEVELOPMENT.md) for the separate browser validation.
> The hardware limitations below still apply.

## Decision

**Keep NanoMaestro Light as a candidate for a small, continuously stateful note generator.**
Its original INT8 graph executes on emulated Cortex-A7/ARMv7 with a real 32-bit
ONNX Runtime process. This is not yet a Y2-ready player or proof of real-time
performance. No NanoMaestro model is connected to the synth or GitHub Pages.
FLOW remains paused.

NanoMaestro produces piano-oriented symbolic events (notes, positions, durations,
velocities), not a finished audio stream. Y2Synth would still need an event decoder,
scheduling, synthesis and effects. The 512-step smoke test does not establish
indefinite generation, musical quality, instrument separation or robust handling
of EOS/malformed event sequences.

## Reproducible inputs

- Model: [utkucoban/NanoMaestro-Realtime](https://huggingface.co/utkucoban/NanoMaestro-Realtime).
- Pinned revision: `9a64dd24b3a550a6101f868af71bdc9960376e26`.
- Original `NanoMaestro-Light/model_int8.onnx`: **13,984,822 bytes**
  (13.98 MB / 13.34 MiB).
- SHA-256: `1d8f054bd04bf63d2d0514e6773d5a293c8034b7488684daeb44311efa96a87c`.
- Matching original vocabulary: 38,544 bytes; SHA-256
  `51a3142aa391e9d588e866a87657a792e72f03a0cbc63ad5057760d38e7c67b3`.
- ONNX IR 8, standard opset 17, Microsoft opset 1; includes DynamicQuantizeLSTM,
  DynamicQuantizeLinear and MatMulInteger.
- Hidden/cell state: two float32 tensors of shape [2, 1, 1024], **16 KiB together**.
  This is only recurrent state, not total inference memory.
- Runtime: Debian trixie ONNX Runtime 1.21.0, CPU, one thread, graph optimization
  disabled; C++ compiled for ARMv7 hard-float with Cortex-A7/NEON/VFPv4.
- Execution: `qemu-arm -cpu cortex-a7` in GitHub Actions; Python/x86 CPU reference.
- Exact Debian packages, ELF/ABI metadata and reports are in each run's artifact,
  retained for 14 days. Model weights and executables are not committed.

Official ONNX Runtime Git/codeload requests failed from the runner during this
experiment. Signed Debian packages allowed the ARM32 check to proceed. This is
the full distribution runtime, not a reduced custom runtime.

## Observations

| Measurement | Initial run | Diagnostic run |
| --- | ---: | ---: |
| Reference trajectory steps | 256 | 256 |
| Matching argmax choices | 255/256 | 256/256 |
| Maximum probability difference | 0.0697142 | 0.000000356007 |
| Maximum logit difference | 3.44145 | 0.0000152588 |
| Maximum final recurrent state difference | 0.815868 | 0.00000858307 |
| Independently sampled ARM tokens | 512 | 512 |
| Numerical gate | **FAIL** | **PASS** |

- [Initial run 44810282336](https://github.com/mvpbit/Y2Synth/actions/runs/44810282336)
  at commit `0a5d260dcf027b1ece4223b4dd4107e294c51084`.
- [Diagnostic run 44810973072](https://github.com/mvpbit/Y2Synth/actions/runs/44810973072)
  at commit `3f4c61f2b66e44f89a6e27dfa56f63caa4d2d820`.

Acceptance remained unchanged: all neural outputs finite, maximum softmax
probability difference <= 0.02, >= 90% matching argmax choices.
The diagnostic did not change model weights, the original comparison loop or
the acceptance rule. It added another 256 comparisons that restore the exact
x86 hidden/cell inputs before every step. Those also matched 256/256 argmax
choices, with maximum probability difference 0.000000357932.

**The discrepancy between runs is unresolved.** One successful run is evidence
of execution and agreement on that trajectory, not proof of reproducibility on
all runners. It does not retroactively fix the initial failed comparison.
The runs did not record CPU feature sets or fixture hashes, so the host-dependent
kernel hypothesis cannot be confirmed from these artifacts. ONNX Runtime
[documents architecture-dependent INT8 saturation behavior](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-and-why-do-i-need-to-try-u8u8);
that is a possible investigation direction, not an established cause here.
Before accepting this as a numerical regression baseline, record host CPU
features and fixture hashes and compare a controlled reference backend.

The diagnostic run recorded 23.1833 seconds for its three inference passes
(256 stateful + 256 reset-state + 512 sampled steps), plus comparison/sampling
work, and emulated-process max RSS of 90,584 KiB. **Neither number is Y2 latency
or Y2 RAM usage.** The initial run's 22.8731 seconds covered fewer passes and
must not be compared as a speed regression. The x86 reference's timing also
does not measure the target device.

## Remaining hardware gates

The [Y2 community hardware/app reference](https://github.com/y1-community/Innioasis-Updater/blob/main/skills/android-apps/SKILL.md)
lists MT6582, four Cortex-A7 cores and Android 4.4.2/API 19.
The [manufacturer specification](https://www.innioasis.com/a/docs/y2/specifications)
lists 1 GB RAM. A small weight file alone does not establish fit in the running
system.

1. Choose the device OS/runtime target. This Linux/glibc executable is not an
   Android/Bionic binary and cannot simply be installed as an APK.
2. Resolve the cross-run numerical discrepancy using a controlled reference.
3. On a physical Y2, measure model load peak memory, sustained and tail token
   latency, and inference throughput with the intended synth/FX running.
4. Exercise continuous event decoding, buffering, EOS/recovery, audio underruns
   and a long listening/battery test.

The experiment and report were created entirely through GitHub; all builds and
executions ran in GitHub Actions. No device measurement or new Pages deployment
was performed.
