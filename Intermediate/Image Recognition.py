"""
Image Recognition with Convolutional Neural Networks (CNNs)
=============================================================
Classifies images from a synthetic dataset (or CIFAR-10) using:
- Custom CNN architecture with Conv2D, BatchNorm, MaxPooling, Dropout
- Data augmentation
- Training history visualisation
- Confusion matrix & per-class accuracy
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Check for TensorFlow ───────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("[INFO] TensorFlow not installed.")
    print("       Run: pip install tensorflow")
    print("       Then re-run this script for the full CNN demo.\n")

print("=" * 60)
print("  IMAGE RECOGNITION WITH CNNs")
print("=" * 60)

# ── Use CIFAR-10 if TF available, else synthetic demo ─────────────────────────
if TF_AVAILABLE:
    print("\n[1] Loading CIFAR-10 dataset …")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
    NUM_CLASSES = 10
    IMG_SIZE    = 32

    # Normalise
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── Visualise sample images ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i])
        ax.set_title(CLASS_NAMES[y_train[i][0]], fontsize=10)
        ax.axis("off")
    fig.suptitle("CIFAR-10 Sample Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=120, bbox_inches="tight")
    print("  Sample images saved → sample_images.png")

    # ── Data augmentation ──────────────────────────────────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    datagen.fit(X_train)

    # ── CNN model ─────────────────────────────────────────────────────────────
    def build_cnn(input_shape=(32, 32, 3), num_classes=10):
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense head
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ])
        return model

    model = build_cnn()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Training ───────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    print("\n[2] Training CNN …")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=64),
        steps_per_epoch=len(X_train) // 64,
        epochs=30,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ─────────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n[3] Test Accuracy : {test_acc:.4f}")
    print(f"    Test Loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = y_test.flatten()

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    print("\nPer-class Accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        cls_acc = cm[i, i] / cm[i].sum()
        print(f"  {name:12s}: {cls_acc:.1%}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    import seaborn as sns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CNN Training Results", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy"); axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss"); axes[1].legend()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[2].set_title("Confusion Matrix")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("cnn_results.png", dpi=150, bbox_inches="tight")
    print("Plots saved → cnn_results.png")

    model.save("cifar10_cnn_model.h5")
    print("Model saved → cifar10_cnn_model.h5")

else:
    # ── Architecture demo without TF ──────────────────────────────────────────
    print("""
CNN Architecture (requires TensorFlow to train):

Input: (32, 32, 3)
│
├─ Conv2D(32, 3×3) → BN → ReLU
├─ Conv2D(32, 3×3) → BN → ReLU
├─ MaxPool(2×2) → Dropout(0.25)
│
├─ Conv2D(64, 3×3) → BN → ReLU
├─ Conv2D(64, 3×3) → BN → ReLU
├─ MaxPool(2×2) → Dropout(0.25)
│
├─ Conv2D(128, 3×3) → BN → ReLU
├─ MaxPool(2×2) → Dropout(0.25)
│
├─ Flatten
├─ Dense(256) → BN → ReLU → Dropout(0.5)
└─ Dense(10, softmax)   [CIFAR-10 classes]

Install TensorFlow to run full training:
  pip install tensorflow
""")

print("\nCNN project complete!")