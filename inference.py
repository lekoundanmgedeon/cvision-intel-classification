"""
inference.py – Load a saved model and run predictions on new images.

Usage examples
──────────────
# PyTorch
python inference.py --framework pytorch \
    --model_path outputs/your_firstname_model.pth \
    --image_path sample.jpg

# TensorFlow
python inference.py --framework tensorflow \
    --model_path outputs/your_firstname_model.keras \
    --image_path sample.jpg
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE    = 150

# ── Normalisation constants (ImageNet) ────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr  # (H, W, 3)  float32


def predict_pytorch(model_path: str, img_path: str):
    import torch
    from model.pytorch_model import IntelCNN

    ckpt   = torch.load(model_path, map_location="cpu", weights_only=False)
    model  = IntelCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    arr  = preprocess_image(img_path)           # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    return probs


def predict_tensorflow(model_path: str, img_path: str):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    arr   = preprocess_image(img_path)          # (H, W, 3)
    batch = np.expand_dims(arr, axis=0)         # (1, H, W, 3)
    probs = model.predict(batch, verbose=0)[0]
    return probs


def show_prediction(img_path: str, probs: np.ndarray):
    img = Image.open(img_path).convert("RGB")
    top_idx = np.argmax(probs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img)
    axes[0].set_title(f"Predicted: {CLASS_NAMES[top_idx]}  ({probs[top_idx]*100:.1f}%)")
    axes[0].axis("off")

    colors = ["steelblue"] * len(CLASS_NAMES)
    colors[top_idx] = "tomato"
    axes[1].barh(CLASS_NAMES, probs * 100, color=colors)
    axes[1].set_xlabel("Probability (%)")
    axes[1].set_title("Class probabilities")
    axes[1].set_xlim(0, 100)
    for i, v in enumerate(probs):
        axes[1].text(v * 100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    out = "prediction_result.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Result plot saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Intel Image Classifier – Inference")
    parser.add_argument("--framework",  required=True, choices=["pytorch", "tensorflow"])
    parser.add_argument("--model_path", required=True, help="Path to saved model file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        sys.exit(f"Image not found: {args.image_path}")
    if not os.path.isfile(args.model_path):
        sys.exit(f"Model not found: {args.model_path}")

    if args.framework == "pytorch":
        probs = predict_pytorch(args.model_path, args.image_path)
    else:
        probs = predict_tensorflow(args.model_path, args.image_path)

    top_idx = np.argmax(probs)
    print(f"\nPredicted class : {CLASS_NAMES[top_idx]}  ({probs[top_idx]*100:.1f}%)")
    print("\nAll class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        bar = "█" * int(p * 40)
        print(f"  {name:12s} {p*100:5.1f}%  {bar}")

    show_prediction(args.image_path, probs)


if __name__ == "__main__":
    main()
