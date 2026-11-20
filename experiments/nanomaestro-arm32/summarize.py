"""Summarize the isolated probe; never extrapolate QEMU timing to the Y2."""
import json
import os
from pathlib import Path
root = Path("nano-results")
manifest = json.loads((root/"manifest.json").read_text())
report_path = root/"arm32-report.json"
report = json.loads(report_path.read_text()) if report_path.exists() else None
lines = ["# NanoMaestro Light ARM32 feasibility probe", "",
    "This run tests Linux ARMv7/Cortex-A7 instructions under QEMU, not an Innioasis Y2.",
    "No hardware speed, peak RAM, battery life, audio quality or realtime result is implied.", "",
    "Model revision: `"+manifest["revision"]+"`.",
    "Runtime: ONNX Runtime 1.23.2, CPU, one thread, graph optimizations disabled.",
    "", "## Model", "```json", json.dumps(manifest,indent=2), "```"]
if report:
    lines += ["", "## ARM32 functional result", "```json", json.dumps(report,indent=2), "```"]
    vocab=json.loads(Path("nano-model/vocab.json").read_text())["itos"]
    ids=(root/"arm32-token-ids.txt").read_text().split()
    events=[vocab[x] for x in ids]
    (root/"arm32-events.txt").write_text(" ".join(events))
    lines += ["", "First sampled event tokens:", "```", " ".join(events[:80]), "```"]
else:
    lines += ["", "ARM32 did not reach a completed inference report. Inspect build/run logs."]
(root/"REPORT.md").write_text("\n".join(lines)+"\n")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:f.write("\n".join(lines)+"\n")
print("\n".join(lines))
