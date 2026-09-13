"""Convert the trained Keras model to a quantized TFLite artifact and sanity-check accuracy."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf


def evaluate_tflite(tflite_path: Path, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    correct_top1 = 0
    correct_top5 = 0
    for i in range(len(X)):
        interpreter.set_tensor(input_detail["index"], X[i : i + 1])
        interpreter.invoke()
        probs = interpreter.get_tensor(output_detail["index"])[0]
        top5 = np.argsort(probs)[-5:]
        if top5[-1] == y[i]:
            correct_top1 += 1
        if y[i] in top5:
            correct_top5 += 1

    n = len(X)
    return correct_top1 / n, correct_top5 / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="D:/sketch-oracle-data/models/sketch_oracle.keras")
    parser.add_argument("--classes", default="D:/sketch-oracle-data/processed/classes.txt")
    parser.add_argument("--test-data", default="D:/sketch-oracle-data/processed/test.npz")
    parser.add_argument("--eval-samples", type=int, default=2000, help="Subset size for the accuracy sanity check")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}")
    model = tf.keras.models.load_model(args.model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # post-training dynamic-range INT8 quantization
    tflite_model = converter.convert()

    out_model = out_dir / "sketch_oracle_quant.tflite"
    out_model.write_bytes(tflite_model)

    orig_size = Path(args.model).stat().st_size
    quant_size = out_model.stat().st_size
    print(f"{args.model}: {orig_size / 1e6:.2f} MB -> {out_model}: {quant_size / 1e6:.2f} MB "
          f"({orig_size / quant_size:.1f}x smaller)")

    shutil.copy(args.classes, out_dir / "classes.txt")

    data = np.load(args.test_data)
    X, y = data["X"], data["y"]
    if args.eval_samples and len(X) > args.eval_samples:
        idx = np.random.default_rng(0).choice(len(X), args.eval_samples, replace=False)
        X, y = X[idx], y[idx]

    top1, top5 = evaluate_tflite(out_model, X, y)
    print(f"quantized model on {len(X)} test samples: top1={top1:.3f}  top5={top5:.3f}")


if __name__ == "__main__":
    main()
