"""
preprocess.py — Fundus image preprocessing utilities.

Key routines:
  - load_image_rgb        : safe BGR→RGB loader
  - circular_crop         : OTSU-based fundus isolation
  - ben_graham_normalize  : Ben Graham local-contrast normalisation
  - clahe_enhance         : CLAHE on L-channel of LAB colourspace
  - histogram_match_rgb   : skimage histogram matching for domain alignment

CHANGELOG (vs original):
  - ben_graham_normalize: FIXED — replaced incorrect z-score normalisation with
    the actual Ben Graham method (4*img - 4*GaussianBlur(img) + 128, clip, /255).
    The original code computed (image - mean) / std which is standard whitening,
    NOT the Graham approach described in the paper (Section 2.2) and the Kaggle
    winning kernel that this method refers to.
"""

import cv2
import numpy as np

try:
    from skimage.exposure import match_histograms
except ImportError:  # pragma: no cover
    match_histograms = None


def load_image_rgb(image_path: str) -> np.ndarray:
    """Read an image from disk and return as uint8 RGB array."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def circular_crop(image_rgb: np.ndarray, target_size: int | None = None) -> np.ndarray:
    """
    Isolate the retinal disc from the dark background using OTSU thresholding,
    fit a minimum enclosing circle to the largest contour, and crop to a square
    bounding box around that circle.  Pads with zeros if the circle extends
    beyond the image boundary.

    Args:
        image_rgb  : uint8 RGB image (H×W×3).
        target_size: if not None, resize the cropped square to this side length.

    Returns:
        Cropped (and optionally resized) uint8 RGB image.
    """
    if image_rgb.ndim != 3:
        raise ValueError("Expected RGB image with 3 dimensions")

    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # OTSU thresholding to isolate fundus from dark background.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fallback: invert mask if no contours found (very dark images).
    if not contours:
        _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Last resort: centre-crop to min(H,W) square.
    if not contours:
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = image_rgb[y0:y0 + side, x0:x0 + side]
        if target_size is not None:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return crop

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), max(int(radius), 1)

    x1, y1 = cx - radius, cy - radius
    x2, y2 = cx + radius, cy + radius

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        image_rgb = cv2.copyMakeBorder(
            image_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        x1 += pad_left;  x2 += pad_left
        y1 += pad_top;   y2 += pad_top

    crop = image_rgb[y1:y2, x1:x2]
    if target_size is not None:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return crop


def ben_graham_normalize(image_rgb_uint8: np.ndarray, sigma_ratio: float = 0.1) -> np.ndarray:
    """
    Ben Graham local-contrast normalisation (Graham, Kaggle 2015).

    Subtracts a heavily Gaussian-blurred copy to remove low-frequency
    illumination gradients, adds 128 to re-centre, and clips to [0,255].
    Divides by 255 to produce float32 in [0,1] ready for ImageNet normalisation.

        output = clip(4 * img - 4 * GaussianBlur(img, sigma) + 128, 0, 255) / 255

    sigma = sigma_ratio * min(H, W)  →  ~10 px at 224 px input (default 0.1).

    FIX (vs original): the original code applied z-score normalisation
    (image - mean) / std, which is standard whitening — NOT the Graham method
    referenced in Section 2.2 of the paper. This is now corrected.

    Args:
        image_rgb_uint8: uint8 RGB image, shape (H, W, 3).
        sigma_ratio    : Gaussian sigma as a fraction of the shorter side.

    Returns:
        float32 RGB image in [0, 1].
    """
    if image_rgb_uint8.dtype != np.uint8:
        image_rgb_uint8 = np.clip(image_rgb_uint8, 0, 255).astype(np.uint8)

    h, w = image_rgb_uint8.shape[:2]
    sigma = max(1, int(sigma_ratio * min(h, w)))
    ksize = sigma * 4 + 1  # must be odd; covers ±2σ on each side

    img_f = image_rgb_uint8.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)

    result = 4.0 * img_f - 4.0 * blurred + 128.0
    result = np.clip(result, 0.0, 255.0) / 255.0
    return result.astype(np.float32)


def clahe_enhance(
    image_rgb_uint8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE applied to the L-channel of the LAB colour space (Section 2.2).

    Args:
        image_rgb_uint8: uint8 RGB image, shape (H, W, 3).
        clip_limit     : CLAHE clip limit (default 2.0 per paper).
        tile_grid_size : tile grid size  (default (8,8) per paper).

    Returns:
        uint8 RGB image with enhanced local contrast.
    """
    lab = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab_enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def histogram_match_rgb(image_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """
    Adjust the colour/intensity distribution of *image_rgb* to match
    *reference_rgb* using per-channel histogram matching (Gonzalez & Woods, 2018).
    Applied only to Messidor-2 images to reduce domain shift vs APTOS (Sec 2.2).

    Args:
        image_rgb    : uint8 RGB source image.
        reference_rgb: uint8 RGB reference image (from APTOS training set).

    Returns:
        uint8 RGB image with matched histogram.
    """
    if match_histograms is None:
        raise ImportError(
            "scikit-image is required for histogram matching. "
            "Install with: pip install scikit-image"
        )
    matched = match_histograms(image_rgb, reference_rgb, channel_axis=-1)
    return np.clip(matched, 0, 255).astype(np.uint8)