"""
Computer Vision Projects
=========================
A collection of computer vision tasks:
1. Image preprocessing pipeline (resize, denoise, edge detection)
2. Object detection simulation (bounding boxes + NMS)
3. Image segmentation (K-means colour segmentation)
4. Feature extraction with HOG descriptors
5. Transfer learning demo (VGG16 feature extractor — requires TF)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

np.random.seed(42)

print("=" * 60)
print("  COMPUTER VISION PROJECTS")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# HELPER: create a synthetic scene image
# ══════════════════════════════════════════════════════════════
def create_synthetic_image(h=128, w=128):
    img = np.ones((h, w, 3), dtype=np.uint8) * 200        # grey background
    # Sky gradient
    for i in range(h // 2):
        img[i, :] = [135 - i, 170 - i // 2, 235 - i // 3]
    # Ground
    img[h // 2:, :] = [80, 120, 60]
    # Sun
    cx, cy, r = 100, 20, 15
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask] = [255, 230, 50]
    # Building
    img[40:100, 20:60] = [150, 150, 160]
    img[40:55,  30:50] = [255, 220, 100]   # windows
    # Tree trunk & leaves
    img[80:100, 85:90] = [100, 60, 20]
    Y2, X2 = np.ogrid[:h, :w]
    tree = (X2 - 87)**2 + (Y2 - 70)**2 <= 18**2
    img[tree] = [34, 139, 34]
    return img.clip(0, 255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════
# 1. IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n[1] Image Preprocessing Pipeline")

def rgb2gray(img):
    return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)

def gaussian_blur(gray, ksize=5, sigma=1.0):
    k  = np.arange(-(ksize//2), ksize//2 + 1)
    g1 = np.exp(-k**2 / (2*sigma**2))
    kernel = np.outer(g1, g1); kernel /= kernel.sum()
    pad = ksize // 2
    padded = np.pad(gray.astype(float), pad, mode="reflect")
    out = np.zeros_like(gray, dtype=float)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            out[i, j] = (padded[i:i+ksize, j:j+ksize] * kernel).sum()
    return out.clip(0, 255).astype(np.uint8)

def sobel_edges(gray):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky = Kx.T
    pad = np.pad(gray.astype(float), 1, mode="reflect")
    Gx = np.zeros_like(gray, dtype=float)
    Gy = np.zeros_like(gray, dtype=float)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            patch = pad[i:i+3, j:j+3]
            Gx[i,j] = (patch * Kx).sum()
            Gy[i,j] = (patch * Ky).sum()
    return np.sqrt(Gx**2 + Gy**2).clip(0, 255).astype(np.uint8)

img      = create_synthetic_image()
gray     = rgb2gray(img)
blurred  = gaussian_blur(gray, ksize=5, sigma=1.5)
edges    = sobel_edges(blurred)
print(f"  Image shape : {img.shape}")
print(f"  Gray range  : [{gray.min()}, {gray.max()}]")
print(f"  Edge pixels > 50: {(edges > 50).sum()}")

# ══════════════════════════════════════════════════════════════
# 2. K-MEANS COLOUR SEGMENTATION
# ══════════════════════════════════════════════════════════════
print("\n[2] K-Means Colour Segmentation")

def kmeans_segment(img, k=4, max_iter=20):
    h, w, c = img.shape
    pixels   = img.reshape(-1, c).astype(float)
    # Init centres randomly
    idx = np.random.choice(len(pixels), k, replace=False)
    centres = pixels[idx].copy()
    labels  = np.zeros(len(pixels), dtype=int)
    for _ in range(max_iter):
        dists  = np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centres[j] = pixels[mask].mean(axis=0)
    segmented = centres[labels].reshape(h, w, c).astype(np.uint8)
    return segmented, labels.reshape(h, w), centres

seg_img, seg_labels, centres = kmeans_segment(img, k=4)
print(f"  Segments found : 4")
print(f"  Dominant colours (RGB):")
for i, c in enumerate(centres.astype(int)):
    print(f"    Cluster {i}: {c}")

# ══════════════════════════════════════════════════════════════
# 3. SIMULATED OBJECT DETECTION WITH NMS
# ══════════════════════════════════════════════════════════════
print("\n[3] Object Detection with Non-Maximum Suppression")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def nms(boxes, scores, iou_thresh=0.4):
    order = np.argsort(scores)[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        rest = order[1:]
        order = [j for j in rest if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

# Simulate raw detections (overlapping proposals)
raw_boxes = [
    [18, 38, 62, 102], [22, 40, 65, 105], [15, 35, 63, 100],   # building cluster
    [72, 52, 102, 100],[75, 55, 105, 102],                       # tree cluster
    [85, 5, 115, 35],  [90, 8, 118, 38],                         # sun cluster
]
raw_scores = [0.95, 0.80, 0.72, 0.88, 0.73, 0.91, 0.65]
labels_det = ["building","building","building","tree","tree","sun","sun"]

keep = nms(raw_boxes, raw_scores)
print(f"  Raw proposals : {len(raw_boxes)}")
print(f"  After NMS     : {len(keep)}")
print("  Final detections:")
for k in keep:
    print(f"    {labels_det[k]:10s}  box={raw_boxes[k]}  score={raw_scores[k]:.2f}")

# ══════════════════════════════════════════════════════════════
# 4. VISUALISATION
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Computer Vision Pipeline", fontsize=15, fontweight="bold")

axes[0,0].imshow(img); axes[0,0].set_title("Original Image"); axes[0,0].axis("off")
axes[0,1].imshow(gray, cmap="gray"); axes[0,1].set_title("Grayscale"); axes[0,1].axis("off")
axes[0,2].imshow(edges, cmap="gray"); axes[0,2].set_title("Sobel Edge Detection"); axes[0,2].axis("off")
axes[1,0].imshow(blurred, cmap="gray"); axes[1,0].set_title("Gaussian Blur"); axes[1,0].axis("off")
axes[1,1].imshow(seg_img); axes[1,1].set_title("K-Means Segmentation (k=4)"); axes[1,1].axis("off")

# Detection viz
axes[1,2].imshow(img)
colors = {"building":"red","tree":"lime","sun":"yellow"}
for k in keep:
    b = raw_boxes[k]; lbl = labels_det[k]; scr = raw_scores[k]
    rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                               linewidth=2, edgecolor=colors.get(lbl,"white"), facecolor="none")
    axes[1,2].add_patch(rect)
    axes[1,2].text(b[0], b[1]-3, f"{lbl} {scr:.2f}", color=colors.get(lbl,"white"),
                   fontsize=8, fontweight="bold")
axes[1,2].set_title("Object Detection + NMS"); axes[1,2].axis("off")

plt.tight_layout()
plt.savefig("computer_vision_results.png", dpi=150, bbox_inches="tight")
print("\nPlots saved → computer_vision_results.png")

# ══════════════════════════════════════════════════════════════
# 5. TRANSFER LEARNING NOTE
# ══════════════════════════════════════════════════════════════
print("""
[4] Transfer Learning (optional — requires TensorFlow)

Example: VGG16 feature extractor
─────────────────────────────────
  from tensorflow.keras.applications import VGG16
  from tensorflow.keras.models import Model

  base  = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  model = Model(inputs=base.input, outputs=base.output)

  features = model.predict(preprocessed_images)  # shape: (N, 7, 7, 512)

  Install: pip install tensorflow
""")

print("Computer vision project complete!")