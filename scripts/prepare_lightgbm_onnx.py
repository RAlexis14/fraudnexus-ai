
"""Prepare the validated LightGBM ONNX candidate for batch inference."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


CANDIDATE_DIR = Path("models/candidates/lightgbm_weights")
SOURCE = CANDIDATE_DIR / "model_margin_diagnostic.onnx"
OUTPUT = CANDIDATE_DIR / "model_margin_final.onnx"


def main():
    """Correct the ONNX label output batch dimension."""
    model = onnx.load(str(SOURCE))

    # The model must retain the raw-margin representation.
    trees = [
        node for node in model.graph.node
        if node.op_type == "TreeEnsembleClassifier"
    ]

    if len(trees) != 1:
        raise RuntimeError("Expected one ONNX tree ensemble.")

    attributes = {
        attribute.name: attribute
        for attribute in trees[0].attribute
    }

    if attributes["post_transform"].s != b"NONE":
        raise RuntimeError("Expected untransformed raw margins.")

    # Correct batch dimension only; do not modify the trees.
    label_outputs = [
        output for output in model.graph.output
        if output.name == "label"
    ]

    if len(label_outputs) != 1:
        raise RuntimeError("Expected one label output.")

    shape = label_outputs[0].type.tensor_type.shape

    if not shape.dim:
        raise RuntimeError("Label output has no batch dimension.")

    shape.dim[0].ClearField("dim_value")
    shape.dim[0].dim_param = "batch"

    onnx.checker.check_model(model)
    onnx.save(model, str(OUTPUT))

    session = ort.InferenceSession(
        str(OUTPUT),
        providers=["CPUExecutionProvider"],
    )

    print("ONNX final candidate:", OUTPUT)

    for output in session.get_outputs():
        print("Output:", output.name, output.shape, output.type)

    features = np.zeros((4, 15), dtype=np.float32)

    result = session.run(
        None,
        {session.get_inputs()[0].name: features},
    )

    print("Batch test:", len(result), "outputs")
    print("PASS: ONNX accepts a batch of 4 transactions.")


if __name__ == "__main__":
    main()
