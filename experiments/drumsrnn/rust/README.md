# Y2Synth / Rust audio + DrumsRNN experiment

**Current status — 2026-11-17.** This crate is the working **Mac Studio prototype** of Y2Synth, **not** Innioasis Y2 firmware. Its scope grew beyond the initial weights loader: live DrumsRNN, Synth 1 (Pluck/Pad), a procedural melody composer, the sequencer, project state and **shared Master Delay → Reverb** are all in the Rust code.

The user has **auditioned Synth 1** and then the **updated stronger master FX** on Mac, reporting that the sound is now satisfactory. A complete recent `cargo test --locked --features live-audio` log, xrun/CPU benchmarks and any test on the physical Y2 are still **not documented**. Development is paused until the device arrives.

## What is implemented

| Area | Implementation | Verified status |
|---|---|---|
| Checkpoint loader | `lib.rs`, TF.js config/manifest/shards, float32 validation | Mac |
| DrumsRNN | `lstm.rs` + `attention.rs` + `generation.rs`: 3 × 256 BasicLSTMCell, attention length 32, 512 drum classes | Autonomous generation and two numerical PASS comparisons on Mac |
| Drum synthesis | `audio.rs`, `kit.rs`, `dsp.rs` | WAV and live sound on Mac |
| Synth 1 | `synth1.rs`: physical-model-like Pluck, evolving additive Pad, note velocity/length/pitch, track level | **Heard and approved on Mac** |
| Melody composition | `melody.rs`: deterministic seeded motifs/chord progressions; can read existing drums as rhythmic cues | Implemented, separate dedicated listening report not recorded; **not trained ML** |
| Sequencing / saved state | `sequencer.rs`, `project.rs`, `application.rs`, `ui.rs` | Live Mac prototype; last snapshot lives **only in RAM** |
| Master inserts | `master_fx.rs`: post-mix Delay → Reverb, common parameters in `MusicProject` | **Improved version auditioned and approved on Mac** |
| Audio backend | `realtime.rs` / `bin/live.rs`; CPAL/CoreAudio + triple buffer | Mac live works; no production real-time guarantee |

**Not implemented:** Synth 2, Generate All, on-device UI, persistent project files, Y2Linux deployment, stereo FX, AI/ML model for Synth 1 melody generation. Euclidean pitch/duration sequences were discussed, **not integrated**.

## Run on Mac

From the repository root:

```bash
git pull --ff-only
# Only if the local Magenta checkpoint isn't present:
bash experiments/drumsrnn/download-model.sh

cargo test --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --features live-audio

cargo run --release --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --features live-audio --bin live -- \
  --bpm 120 --temperature 1.2
```

The program needs the downloaded local `drum_kit_rnn_tfjs` checkpoint; model weight files are **never committed**.

### Live CLI

Within the `y2>` prompt:

```text
play
regenerate
bpm 100
synth1 pluck
synth1 generate
synth1 pad
synth1 generate
synth1 show
fx
fx off
fx on
fx delay steps 4
fx delay feedback 0.55
fx delay wet 1.0
fx reverb size 0.9
fx reverb damping 0.5
fx reverb wet 1.0
fx reset
status
stop
quit
```

Other commands: `synth1 level 0.7`, `synth1 note 2 72 2`, `synth1 clear`, `kit`, `level kick 0.6`, `tone snare 0.8`, `decay chh 1.5`; wheel simulation `menu`, `center`, `next`, `prev`, `back`, `wheel-play`.

**Master chain:** all drums and synth voices first mix into **one mono bar**, which is then processed by **Delay → Reverb**; these are not per-track inserts. Both effects enabled by default. Delay uses `steps` (1–8 sixteenth notes at the current BPM), feedback (0…0.75), wet (0…1.0). Reverb uses size/damping (0…1.0) and wet (0…1.0). `fx off` bypasses both without discarding their parameters, `fx on` restores them, and `fx reset` restores defaults.

The original effect mix was too weak. Wet is now mixed **in parallel with the direct signal**, and Reverb reflections are stronger. The user confirmed this corrected version sounds good. Output peak protection targets 0.85. FX use an offline 12-bar pre-roll to wrap delay/reverb feedback into periodic loops. All render and model work stays on the control thread: **no FX, LSTM inference or complete-bar synthesis in the CPAL callback**.

During playback, buffer and BPM changes are accepted at the next bar boundary. Project settings and the last rendered content are saved in **RAM only** and disappear on exit.

## Autonomous DrumsRNN / WAV preview

The loader reads `config.json` and `weights_manifest.json` plus referenced binary shards, validates tensor shapes, byte counts and finite values. The checkpoint has **16 tensors, 2,967,850 float32 numbers, 11,871,400 decoded weight bytes (~11.32 MiB)**. The model uses attention length 32 and nine drum MIDI pitches `[36,38,42,46,45,48,50,49,51]` encoded as 512 bitmask classes.

Autonomous DrumsRNN uses one **internal silent bootstrap** with no user-authored groove:

```bash
cargo run --release --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin generate -- --steps 16 --temperature 1.2 \
  --seed 1234 --bpm 120 --wav /tmp/y2synth-groove.wav

afplay /tmp/y2synth-groove.wav
```

`--argmax` is for deterministic verification, not a guarantee of musical sound. Temperature-based random sampling need not match JS event-for-event because the RNG implementations differ.

## Numerical verification against Magenta.js

**Both comparisons passed on Mac Studio (2026-11-17)** with `@magenta/music@1.23.1` / TF.js WebGL and `drum_kit_rnn_tfjs`. Absolute tolerance per probability: `1e-5`.

| Comparison | Probabilities within tolerance | Max absolute error | Different Argmax classes |
|---|---|---|---|
| From scratch / Argmax (silent trajectory) | 8192 / 8192 | `1.192092896e-6` | 0 / 16 |
| Teacher-forced fixed **nonzero** input history | 8192 / 8192 | `1.072883606e-6` | 0 / 16 |

The first comparison predicts 16 silent classes; the second predicts nonzero ones. This establishes strong numerical agreement **on the tested trajectories**, not bit-exact equality across every possible model state.

From the repository root start the browser reference server:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

For silent Argmax: open `http://127.0.0.1:8765/experiments/drumsrnn/reference.html`, select From scratch / Argmax, generate, save JSON. Then:

```bash
cargo run --release --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin verify_reference -- \
  --reference "$HOME/Downloads/drumsrnn-reference.json"
```

For the nonzero diagnostic: open `http://127.0.0.1:8765/experiments/drumsrnn/teacher-forcing.html`, select Generate 16 × 512 reference and Save JSON. Then:

```bash
cargo run --release --locked \
  --manifest-path experiments/drumsrnn/rust/Cargo.toml \
  --bin verify_reference -- \
  --reference "$HOME/Downloads/drumsrnn-teacher-forced.json"
```

These checks are **diagnostic**: the forced input classes are not user performances, and normal `generation.rs` remains autonomous.

## Device bring-up — pending arrival of Innioasis Y2

The physical Y2 has not been characterized. After it arrives, investigate SoC/ABI, available Linux and audio backend, memory, CPU cost of DrumsRNN and the 12-pass FX pre-render, xrun/latency, wheel/display interface and project file persistence. No claim of embedded compatibility or production real-time safety is justified yet.

See [project README](../../../README.md), [ARCHITECTURE](../../../docs/ARCHITECTURE.md), [ROADMAP](../../../docs/ROADMAP.md) and [DEVELOPMENT](../../../docs/DEVELOPMENT.md).
