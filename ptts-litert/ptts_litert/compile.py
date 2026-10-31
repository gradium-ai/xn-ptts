"""Compile a bundle's model ahead of time for NPUs, one `.tflite` per SoC.

Only `prefill`, `flow_step` and `mimi_step` are offered to the NPU compiler; the two table
signatures stay on the CPU. For each target this prints how many of each signature's ops
landed on the NPU and in how many pieces: one piece per signature is what to aim for, since
every boundary is a round trip between the CPU and the NPU.

The vendor compilers run on Linux x86_64 only.
"""

import json
import shutil
import tempfile
from pathlib import Path

NPU_SIGNATURES = ["prefill", "flow_step", "mimi_step"]


def target(spec: str):
    from ai_edge_litert.aot.vendors.mediatek import target as mediatek
    from ai_edge_litert.aot.vendors.qualcomm import target as qualcomm

    vendors = {"qualcomm": qualcomm, "mediatek": mediatek}
    try:
        from ai_edge_litert.aot.vendors.google_tensor import target as google
        from ai_edge_litert.aot.vendors.samsung import target as samsung

        vendors |= {"google": google, "samsung": samsung}
    except ImportError:
        pass
    vendor, _, soc = spec.partition(":")
    if vendor not in vendors or not soc:
        raise SystemExit(f"{spec!r}: expected VENDOR:SOC with VENDOR one of {sorted(vendors)}")
    mod = vendors[vendor]
    try:
        return mod.Target(mod.SocModel(soc))
    except ValueError:
        raise SystemExit(f"{spec!r}: SoCs for {vendor} are {[s.value for s in mod.SocModel if s.value != 'ALL']}")


def subgraphs(path: Path, names: list[str]) -> list[int]:
    from ai_edge_litert import schema_py_generated as schema

    m = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    index = {
        m.SignatureDefs(i).SignatureKey().decode(): m.SignatureDefs(i).SubgraphIndex()
        for i in range(m.SignatureDefsLength())
    }
    return [index[n] for n in names]


def compile(args):
    from ai_edge_litert.aot import aot_compile

    model = args.bundle / "model.tflite"
    out = args.bundle / "npu"
    out.mkdir(exist_ok=True)
    targets = [target(t) for t in args.targets]
    with tempfile.TemporaryDirectory() as tmp:
        result = aot_compile.aot_compile(
            str(model),
            output_dir=tmp,
            target=targets,
            keep_going=True,
            subgraphs_to_compile=subgraphs(model, NPU_SIGNATURES),
        )
        print(result.compilation_report())
        compiled = {}
        for p in sorted(Path(tmp).glob("*.tflite")):
            if p.stat().st_size == 0:
                continue
            name = p.name.removeprefix("model_").removesuffix("_apply_plugin.tflite")
            shutil.move(p, out / f"{name}.tflite")
            compiled[name] = f"npu/{name}.tflite"
    if not compiled:
        raise SystemExit("nothing compiled")
    # Recorded in bundle.json, so a runtime can pick the file for the SoC it runs on.
    meta_path = args.bundle / "bundle.json"
    meta = json.loads(meta_path.read_text())
    meta["npu"] = {**meta.get("npu", {}), **compiled}
    meta_path.write_text(json.dumps(meta, indent=2))
    for name, path in compiled.items():
        print(f"wrote {args.bundle / path}")
