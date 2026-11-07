"""Tests the 5 classical-CV operations on synthetic images. Run: python test_cv.py
No GPU or model needed. Each case should trigger the operation named in EXPECT."""
import sys, os
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classical.postprocess import postprocess

rng = np.random.default_rng(0)

def base(size=512):
    """Smooth gradient + a few shapes, so edges exist but the image is clean."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.stack([60 + 140 * x / size, 80 + 120 * y / size, 200 - 120 * x / size], -1)
    cv2.circle(img, (size // 3, size // 3), size // 6, (230, 230, 60), -1)
    cv2.rectangle(img, (size // 2, size // 2), (size - 60, size - 60), (40, 200, 90), -1)
    return np.clip(img, 0, 255).astype(np.uint8)

def to_pil(a):
    return Image.fromarray(a)

def noisy(a, sigma=18):
    return np.clip(a.astype(np.float32) + rng.normal(0, sigma, a.shape), 0, 255).astype(np.uint8)

def grid(a, period=8, amp=40):
    g = a.astype(np.float32).copy()
    g[::period, :, :] += amp
    g[:, ::period, :] += amp
    return np.clip(g, 0, 255).astype(np.uint8)

def clipped():
    """Neutral-colour image with ~35% pure-white and ~10% pure-black pixels."""
    g = cv2.cvtColor(base(), cv2.COLOR_BGR2GRAY)
    g = np.where(g > 150, 255, np.where(g < 70, 0, g)).astype(np.uint8)
    return np.stack([g, g, g], -1)

cases = {
    "small_256 (resize)":        (to_pil(cv2.resize(base(), (256, 256))), "resized"),
    "colour_cast (white bal.)":  (to_pil(np.clip(base().astype(np.float32) * [1.0, 0.7, 1.6], 0, 255).astype(np.uint8)), "colour"),
    "low_contrast (clahe)":      (to_pil((base().astype(np.float32) * 0.25 + 95).astype(np.uint8)), "colour"),
    "noisy (edge-aware)":        (to_pil(noisy(base())), "edge_aware"),
    "clipped (tone curve)":      (to_pil(clipped()), "artifacts_found:clipping"),
    "blurry (unsharp)":          (to_pil(cv2.GaussianBlur(base(), (0, 0), 6)), "artifacts_found:blur"),
    "grid_pattern (median)":     (to_pil(grid(base())), "artifacts_found:grid_pattern"),
    "black_frame (reported)":    (to_pil(np.zeros((512, 512, 3), np.uint8)), "artifacts_found:black_frame"),
    "clean (nothing needed)":    (to_pil(base()), None),
}

triggered = {"resize": False, "colour": False, "edge_aware": False, "quality_score": False, "artifacts": False}
print(f"{'case':28} {'expected':30} {'ok':3}  q_before -> q_after  kept")
print("-" * 95)
for name, (img, expect) in cases.items():
    out, rep = postprocess(img)
    assert out.size == (512, 512), f"{name}: output is {out.size}, expected 512x512"
    if rep["resized"]: triggered["resize"] = True
    if rep["colour"]: triggered["colour"] = True
    if rep["edge_aware"]: triggered["edge_aware"] = True
    if rep["artifacts_found"]: triggered["artifacts"] = True
    if "quality_before" in rep and "quality_after" in rep: triggered["quality_score"] = True
    if expect is None:
        ok = True
    elif ":" in expect:
        key, val = expect.split(":"); ok = val in rep[key]
    else:
        ok = bool(rep[expect])
    print(f"{name:28} {str(expect):30} {'OK' if ok else 'NO':3}  {rep['quality_before']:.4f} -> {rep['quality_after']:.4f}   {rep['correction_kept']}")
    print(f"{'':28} colour={rep['colour']} edge={rep['edge_aware']} found={rep['artifacts_found']} fixed={rep['artifacts_fixed']}")

print("\nOperations triggered at least once:")
for k, v in triggered.items():
    print(f"  {k:14} {'YES' if v else 'NO  <-- needs a look'}")
print("\nALL 5 OPERATIONS TRIGGERED" if all(triggered.values()) else "\nSOME OPERATIONS DID NOT TRIGGER")
