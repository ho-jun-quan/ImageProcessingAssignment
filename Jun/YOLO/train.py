"""
train.py — Train a YOLO11 detector on the cloud-chamber particle-track dataset.

Assumes prepare_dataset.py has already produced dataset/data.yaml. Training runs
on the ROCm GPU by default (config.DEVICE = 0) and automatically falls back to
CPU if a ROCm/MIOpen kernel-compilation error is hit. Outputs land in
runs/<PROJECT_NAME>/<RUN_NAME>/, and the best weights are copied to
config.BEST_WEIGHTS.

Run directly:
    python train.py                # use config.DEVICE (GPU, CPU fallback)
    python train.py --device cpu   # force CPU
    python train.py --device 0 --epochs 100
"""

import argparse
import os

from ultralytics import YOLO

import config


def train(model_name=None, epochs=None, imgsz=None, batch=None, device=None):
    """
    Fine-tune a pretrained YOLO11 model on the particle-track dataset.

    Returns the Ultralytics results object.
    """
    model_name = model_name or config.BASE_MODEL
    epochs = epochs if epochs is not None else config.EPOCHS
    imgsz = imgsz if imgsz is not None else config.IMG_SIZE
    batch = batch if batch is not None else config.BATCH
    device = device if device is not None else config.DEVICE

    data_yaml = config.resolve_training_data_yaml()
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(
            f"No YOLO dataset found. Expected either {config.VIDEO_DATA_YAML} or {config.DATA_YAML}. "
            "Run the video-derived preparation script or prepare_dataset.py first."
        )

    print(f"Training on dataset: {data_yaml}")

    try:
        return _run_training(model_name, epochs, imgsz, batch, device, data_yaml)
    except Exception as exc:  # noqa: BLE001
        if device != "cpu" and _is_gpu_kernel_error(exc):
            print(
                "\n[warn] GPU training failed with a ROCm/MIOpen kernel error:\n"
                f"       {type(exc).__name__}: {exc}\n"
                "[warn] Retrying on CPU (device='cpu')...\n"
            )
            return _run_training(model_name, epochs, imgsz, batch, "cpu", data_yaml)
        raise


def _is_gpu_kernel_error(exc):
    """
    True if the exception looks like the ROCm/MIOpen runtime kernel-compilation
    failure seen on some Windows RDNA builds (so we can fall back to CPU).
    """
    msg = str(exc).lower()
    needles = ("miopen", "hiprtc", "hip error", "no kernel image", "type_traits")
    return any(n in msg for n in needles)


def _run_training(model_name, epochs, imgsz, batch, device, data_yaml):
    """Fit a fresh YOLO model on the given device and export best weights."""
    # On CPU, mixed precision and multi-worker mosaic hurt more than help.
    on_cpu = str(device) == "cpu"
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0 if on_cpu else 8,
        amp=not on_cpu,
        seed=config.SEED,
        patience=config.PATIENCE,
        project=os.path.join(config.RUNS_DIR, config.PROJECT_NAME),
        name=config.RUN_NAME,
        exist_ok=True,
        # Augmentation tuned for a tiny dataset of tall, faint tracks:
        degrees=15.0,        # small rotations — tracks have no canonical up
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,          # cloud-chamber tracks are orientation-agnostic
        mosaic=1.0,
        hsv_v=0.4,           # brightness jitter (exposure varies frame to frame)
        hsv_s=0.0,           # tracks are near-greyscale; leave saturation alone
        hsv_h=0.0,
    )

    _export_best_weights(model)
    return results


def _export_best_weights(model):
    """Copy the run's best.pt to config.BEST_WEIGHTS for easy reuse."""
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None) if trainer else None
    if best and os.path.isfile(best):
        os.makedirs(os.path.dirname(config.BEST_WEIGHTS), exist_ok=True)
        if os.path.abspath(best) != os.path.abspath(config.BEST_WEIGHTS):
            import shutil
            shutil.copy2(best, config.BEST_WEIGHTS)
        print(f"Best weights available at: {config.BEST_WEIGHTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO particle-track detector.")
    parser.add_argument("--device", default=None,
                        help="'cpu', '0', etc. Defaults to config.DEVICE with CPU fallback.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--model", default=None, help="Base weights, e.g. yolo11n.pt")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
