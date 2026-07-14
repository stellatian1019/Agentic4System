from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_manifest(input_dir):
    input_dir = Path(input_dir)

    with open(
        input_dir / "manifest.json",
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    tensors = {}

    for item in manifest["tensors"]:
        name = item["name"]
        file = item["file"]

        tensors[name] = np.load(
            input_dir / file
        )

    return manifest, tensors



def save_output(
    output_dir,
    outputs,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensors = []

    for name, array in outputs.items():

        filename = f"{name}.npy"

        np.save(
            output_dir / filename,
            array.astype(np.float32),
        )

        tensors.append(
            {
                "name": name,
                "file": filename,
                "dtype": "float32",
                "shape": list(array.shape),
            }
        )


    manifest = {
        "tensors": tensors
    }

    with open(
        output_dir / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )