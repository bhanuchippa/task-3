"""
train_model.py — Complete CIFAR-10 Training Pipeline using MobileNetV2
======================================================================
Two-phase transfer-learning approach:
  Phase 1  – Feature extraction (frozen base, high LR)
  Phase 2  – Fine-tuning (partially-unfrozen base, low LR)

Generates accuracy / loss graphs, a confusion-matrix heatmap, and a
JSON training-history file, all saved to the project's 'outputs' folder.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use("Agg")                       # headless backend – no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ──────────────────────────────────────────────────────────────────────
# GPU memory-growth (prevents TF from grabbing all VRAM at once)
# ──────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] GPU(s) detected: {[g.name for g in gpus]}")
else:
    print("[INFO] No GPU detected – training will run on CPU.")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
IMG_SIZE       = 96
BATCH_SIZE     = 64
PHASE1_EPOCHS  = 10
PHASE2_EPOCHS  = 15

CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

# Paths – all relative to *this* script's directory
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "mobilenet_model.h5")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "..", "outputs")

# ──────────────────────────────────────────────────────────────────────
# Data augmentation layer (applied **only** when training=True)
# ──────────────────────────────────────────────────────────────────────
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ],
    name="data_augmentation",
)


# ──────────────────────────────────────────────────────────────────────
# Helper: build the model
# ──────────────────────────────────────────────────────────────────────
def build_model(base_trainable: bool = False) -> keras.Model:
    """Build MobileNetV2-based classifier with optional frozen base.

    The augmentation layer uses the ``training`` flag that Keras passes
    automatically so that augmentation is active only during `model.fit`.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = base_trainable

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # Augmentation – only applied during training
    x = data_augmentation(inputs)              # training flag propagated by Keras

    x = base_model(x, training=False)          # always run BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cifar10_mobilenetv2")
    return model, base_model


# ──────────────────────────────────────────────────────────────────────
# Helper: plot & save figures
# ──────────────────────────────────────────────────────────────────────
def save_accuracy_graph(history: dict, path: str) -> None:
    """Save accuracy-vs-epoch plot (dark theme)."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    ax.plot(history["accuracy"],     label="Training Accuracy",   color="#00d2ff", linewidth=2)
    ax.plot(history["val_accuracy"], label="Validation Accuracy", color="#ff6b6b", linewidth=2)

    ax.set_title("Training & Validation Accuracy", fontsize=16, color="white", pad=15)
    ax.set_xlabel("Epoch", fontsize=13, color="white")
    ax.set_ylabel("Accuracy", fontsize=13, color="white")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(path)}")


def save_loss_graph(history: dict, path: str) -> None:
    """Save loss-vs-epoch plot (dark theme)."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    ax.plot(history["loss"],     label="Training Loss",   color="#00d2ff", linewidth=2)
    ax.plot(history["val_loss"], label="Validation Loss", color="#ff6b6b", linewidth=2)

    ax.set_title("Training & Validation Loss", fontsize=16, color="white", pad=15)
    ax.set_xlabel("Epoch", fontsize=13, color="white")
    ax.set_ylabel("Loss", fontsize=13, color="white")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(path)}")


def save_confusion_matrix(y_true, y_pred, path: str) -> None:
    """Save confusion-matrix heatmap (dark theme)."""
    plt.style.use("dark_background")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor="#2a2a4a",
    )

    ax.set_title("Confusion Matrix", fontsize=16, color="white", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=13, color="white")
    ax.set_ylabel("True Label", fontsize=13, color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {os.path.basename(path)}")


# ──────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Ensure output directory exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── 1. Load CIFAR-10 ────────────────────────────────────────────
    print("Loading CIFAR-10...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Labels are shape (N, 1) – squeeze to (N,)
    y_train_full = y_train_full.squeeze()
    y_test = y_test.squeeze()

    # ── 2. Normalise to float32 [0, 1] then resize ─────────────────
    print("Resizing images...")
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test       = x_test.astype(np.float32) / 255.0

    # Resize 32×32 → 96×96
    x_train_full = tf.image.resize(x_train_full, (IMG_SIZE, IMG_SIZE)).numpy()
    x_test       = tf.image.resize(x_test,       (IMG_SIZE, IMG_SIZE)).numpy()

    # MobileNetV2 preprocessing (scales [0, 1] → [-1, 1])
    x_train_full = preprocess_input(x_train_full * 255.0)  # preprocess_input expects [0,255]
    x_test       = preprocess_input(x_test * 255.0)

    # ── 3. Train / validation split (80 / 20) ──────────────────────
    split = int(0.8 * len(x_train_full))
    x_train, x_val = x_train_full[:split], x_train_full[split:]
    y_train, y_val = y_train_full[:split], y_train_full[split:]

    print(f"  Training samples:   {len(x_train)}")
    print(f"  Validation samples: {len(x_val)}")
    print(f"  Test samples:       {len(x_test)}")

    # ── 4. Build model ──────────────────────────────────────────────
    print("Building model...")
    model, base_model = build_model(base_trainable=False)
    model.summary(print_fn=lambda line: print("  " + line))

    # ── 5. Phase 1 – Feature Extraction ─────────────────────────────
    print("\nPhase 1: Feature Extraction...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase1_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
    ]

    history_phase1 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=PHASE1_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    # ── 6. Phase 2 – Fine-tuning ────────────────────────────────────
    print("\nPhase 2: Fine-tuning...")
    base_model.trainable = True

    # Freeze all layers except the last 30
    freeze_until = len(base_model.layers) - 30
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Base-model layers total: {len(base_model.layers)}")
    print(f"  Trainable layers:        {trainable_count}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase2_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    history_phase2 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=PHASE2_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # ── 7. Merge histories from both phases ─────────────────────────
    combined_history: dict[str, list] = {}
    for key in history_phase1.history:
        combined_history[key] = (
            history_phase1.history[key] + history_phase2.history[key]
        )

    # ── 8. Evaluate on test set ─────────────────────────────────────
    print("\nEvaluating...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    print(f"\n  Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")

    # Predictions
    y_pred_probs = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ── 9. Generate & save graphs ───────────────────────────────────
    print("Generating graphs...")
    save_accuracy_graph(
        combined_history,
        os.path.join(OUTPUTS_DIR, "accuracy_graph.png"),
    )
    save_loss_graph(
        combined_history,
        os.path.join(OUTPUTS_DIR, "loss_graph.png"),
    )
    save_confusion_matrix(
        y_test,
        y_pred,
        os.path.join(OUTPUTS_DIR, "confusion_matrix.png"),
    )

    # ── 10. Save training history as JSON ───────────────────────────
    history_path = os.path.join(OUTPUTS_DIR, "training_history.json")
    # Convert numpy / float32 values so JSON can serialise them
    serialisable = {k: [float(v) for v in vals] for k, vals in combined_history.items()}
    serialisable["test_accuracy"] = float(test_accuracy)
    serialisable["test_loss"]     = float(test_loss)
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"  ✓ Saved training_history.json")

    # ── Done ────────────────────────────────────────────────────────
    print(f"\nModel saved! → {MODEL_SAVE_PATH}")
    print("Training complete!")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
