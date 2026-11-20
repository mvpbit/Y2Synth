"""Download and inspect NanoMaestro Light; create a teacher-forced CPU oracle.
Runs only in the isolated GitHub Actions experiment, never in the synth.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import urllib.parse
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort

OUT = Path("nano-results")
OUT.mkdir(exist_ok=True)
MODEL_DIR = Path("nano-model")
MODEL_DIR.mkdir(exist_ok=True)
REPO = "utkucoban/NanoMaestro-Realtime"


def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


info = json.loads(get("https://huggingface.co/api/models/" + REPO))
revision = "9a64dd24b3a550a6101f868af71bdc9960376e26"
files = {}
for remote, local in [
    ("NanoMaestro-Light/model_int8.onnx", "model.onnx"),
    ("NanoMaestro-Light/vocab.json", "vocab.json"),
]:
    url = "https://huggingface.co/" + REPO + "/resolve/" + revision + "/" + urllib.parse.quote(remote)
    data = get(url)
    (MODEL_DIR / local).write_bytes(data)
    checksum = hashlib.sha256(data).hexdigest()
    expected_hash = {
        "model.onnx": "1d8f054bd04bf63d2d0514e6773d5a293c8034b7488684daeb44311efa96a87c",
        "vocab.json": "51a3142aa391e9d588e866a87657a792e72f03a0cbc63ad5057760d38e7c67b3",
    }[local]
    assert checksum == expected_hash, "Model fixture checksum changed"
    files[remote] = {"bytes": len(data), "sha256": checksum}
model = onnx.load(str(MODEL_DIR / "model.onnx"))
onnx.checker.check_model(model)
assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
opsets = {x.domain or "ai.onnx": x.version for x in model.opset_import}
ops = collections.defaultdict(set)


def visit(graph):
    for node in graph.node:
        ops[node.domain or "ai.onnx"].add(node.op_type)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                visit(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    visit(child)


visit(model.graph)
(OUT / "required_ops.config").write_text(
    "\n".join(f"{domain};{opsets[domain]};{','.join(sorted(names))}"
              for domain, names in sorted(ops.items())) + "\n")
settings = ort.SessionOptions()
settings.intra_op_num_threads = 1
settings.inter_op_num_threads = 1
# Match the reduced ARM runtime's explicit graph; do not add fused operators.
settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), settings,
                               providers=["CPUExecutionProvider"])
inputs = {x.name: x for x in session.get_inputs()}
assert set(inputs) == {"input", "h", "c"}, list(inputs)
assert inputs["h"].shape == inputs["c"].shape
layers, batch, hidden = inputs["h"].shape
assert (batch == 1 or batch is None or isinstance(batch, str)) and layers == 2 and hidden == 1024, inputs["h"].shape
vocab = json.loads((MODEL_DIR / "vocab.json").read_text())
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
prompt = [stoi[x] for x in ["BOS", "BPM_120", "GRID_64", "BAR", "POS_0"]]
h = np.zeros((layers, 1, hidden), dtype=np.float32)
c = np.zeros_like(h)
rng = np.random.default_rng(20261121)
tokens, expected, latency = [], [], []
current = prompt[0]
generated = []


def sample(logits):
    scores = logits.astype(np.float64).reshape(-1) / .9
    if "EOS" in stoi:
        scores[stoi["EOS"]] -= 1
    ids = np.argpartition(scores, -32)[-32:]
    p = np.exp(scores[ids] - scores[ids].max())
    return int(rng.choice(ids, p=p / p.sum()))


# Persist fixed inputs and outputs, so ARM sees the SAME trajectory even if
# floating-point rounding would cause different stochastic choices.
for i in range(256):
    tokens.append(current)
    started = time.perf_counter()
    logits, h, c = session.run(["logits", "h_out", "c_out"],
        {"input": np.array([[current]], dtype=np.int64), "h": h, "c": c})
    latency.append((time.perf_counter() - started) * 1000)
    assert all(np.isfinite(x).all() for x in [logits, h, c])
    expected.append(logits.reshape(-1).astype("<f4"))
    current = prompt[i+1] if i+1 < len(prompt) else sample(logits)
    generated.append(itos[current])
assert np.linalg.norm(h) > 0 and np.linalg.norm(c) > 0
width = expected[0].size
assert width == len(itos)
(OUT / "fixture.txt").write_text(
    f"{len(tokens)} {width} {layers} {hidden} {stoi.get('EOS', -1)}\n" +
    " ".join(map(str, tokens)) + "\n")
np.stack(expected).tofile(OUT / "expected-logits.f32")
np.concatenate([h.reshape(-1), c.reshape(-1)]).astype("<f4").tofile(OUT / "expected-state.f32")
(OUT / "x86-events.txt").write_text(" ".join(generated))
manifest = {
    "model_repository": REPO, "revision": revision, "files": files,
    "license_metadata": info.get("cardData", {}).get("license"),
    "onnx_ir": model.ir_version, "opsets": opsets,
    "operations": {k: sorted(v) for k,v in ops.items()},
    "initializer_types": dict(collections.Counter(
        onnx.TensorProto.DataType.Name(x.data_type) for x in model.graph.initializer)),
    "inputs": {name: {"shape": x.shape, "type": x.type} for name,x in inputs.items()},
    "recurrent_state_bytes": h.nbytes + c.nbytes,
    "reference": {"runtime": ort.__version__, "machine": platform.machine(),
        "steps": len(tokens), "finite": True, "state_norm": float(np.linalg.norm(h)),
        "median_step_ms": float(np.median(latency)), "p95_step_ms": float(np.percentile(latency,95))},
    "limitations": [
        "x86 timing is not Y2 timing.",
        "File size and recurrent state size are not process peak RAM.",
        "QEMU execution is an ISA/functional check, not a hardware benchmark.",
        "This Linux/glibc binary is not a stock Android 4.4 APK or Bionic binary.",
        "No audio-quality, battery, xrun or hardware compatibility claim."
    ]
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
