"""P3 owns this file. The 5 mandatory classical-CV operations, each applied only when a statistic says it is needed."""
import cv2, numpy as np
from PIL import Image

def _stats(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    colourful = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)  # Hasler-Susstrunk
    means = [float(c.mean()) for c in (b, g, r)]
    return dict(
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(lab[..., 0].mean()), contrast=float(lab[..., 0].std()),
        colourfulness=float(colourful), cast=float(max(means) - min(means)),
        clipped=float(((gray <= 2) | (gray >= 253)).mean()),
        noise=float(np.median(np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 3)))))

def quality_score(bgr):
    """4. Image quality assessment: one number in [0,1], higher is better."""
    s = _stats(bgr)
    q = (0.35 * min(s["sharpness"] / 300, 1) + 0.25 * (1 - abs(s["brightness"] - 128) / 128)
         + 0.25 * min(s["contrast"] / 60, 1) + 0.15 * (1 - min(s["noise"] / 8, 1)) - s["clipped"])
    return round(float(np.clip(q, 0, 1)), 4), s

def resize_upscale(bgr, target=512):
    """1. Resizing / resolution optimization: Lanczos + unsharp mask on edges only."""
    if bgr.shape[0] == target and bgr.shape[1] == target:
        return bgr, False
    up = cv2.resize(bgr, (target, target), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    edges = cv2.dilate(cv2.Canny(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY), 80, 160), np.ones((3, 3), np.uint8))
    up[edges > 0] = sharp[edges > 0]
    return up, True

def colour_correct(bgr, s):
    """2. Colour correction and enhancement: only the fixes the statistics ask for."""
    done = []
    if s["cast"] > 25:                                   # gray-world white balance
        f = bgr.astype(np.float32); m = f.reshape(-1, 3).mean(0)
        bgr = np.clip(f * (m.mean() / (m + 1e-6)), 0, 255).astype(np.uint8); done.append("white_balance")
    if s["contrast"] < 40:                               # CLAHE on lightness
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); done.append("clahe")
    if s["colourfulness"] < 20 or s["colourfulness"] > 110:   # saturation
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.15 if s["colourfulness"] < 20 else 0.88), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR); done.append("saturation")
    return bgr, done

def edge_aware(bgr, s):
    """3. Edge-aware post-processing: bilateral filter on flat noisy areas, edges protected."""
    if s["noise"] < 3:
        return bgr, False
    f = cv2.bilateralFilter(bgr, d=5, sigmaColor=30, sigmaSpace=5)
    edges = cv2.dilate(cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 80, 160), np.ones((3, 3), np.uint8))
    f[edges > 0] = bgr[edges > 0]
    return f, True

def artifacts(bgr, s):
    """5. Artifact detection and correction."""
    found, fixed = [], []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 5:
        found.append("black_frame")                      # usually a safety-filter output: reported, not fixable
    if s["clipped"] > 0.02:
        found.append("clipping")
        lut = np.array([255 * ((i / 255) ** 1.1) * 0.97 + 4 for i in range(256)]).clip(0, 255).astype(np.uint8)
        bgr = cv2.LUT(bgr, lut); fixed.append("tone_curve")
    if s["sharpness"] < 40 and "black_frame" not in found:
        found.append("blur")
        blur = cv2.GaussianBlur(bgr, (0, 0), 1.2)
        bgr = cv2.addWeighted(bgr, 1.6, blur, -0.6, 0); fixed.append("unsharp")
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) - gray.mean())))
    h, w = spec.shape; spec[h//2-8:h//2+8, w//2-8:w//2+8] = 0     # ignore low frequencies
    if spec.max() > 40 * np.median(spec) + 1e-6:
        found.append("grid_pattern")
        bgr = cv2.medianBlur(bgr, 3); fixed.append("median_3")
    return bgr, found, fixed

def postprocess(img, target=512):
    """Full pipeline. Returns (output PIL image, report dict). Keeps the uncorrected image if quality does not improve."""
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr, resized = resize_upscale(bgr, target)
    q0, s = quality_score(bgr)
    if cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean() < 5:   # black frame (safety filter): report, do not "enhance"
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), dict(
            resized=resized, colour=[], edge_aware=False, artifacts_found=["black_frame"], artifacts_fixed=[],
            quality_before=q0, quality_after=q0, correction_kept=False)
    out, colour = colour_correct(bgr.copy(), s)
    out, smoothed = edge_aware(out, _stats(out))
    out, found, fixed = artifacts(out, _stats(out))
    q1, _ = quality_score(out)
    accepted = q1 >= q0
    final = out if accepted else bgr
    report = dict(resized=resized, colour=colour, edge_aware=smoothed, artifacts_found=found,
                  artifacts_fixed=fixed, quality_before=q0, quality_after=q1, correction_kept=accepted)
    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB)), report
