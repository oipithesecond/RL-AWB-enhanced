import numpy as np
import cv2

def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction: output = input^(1/gamma)
    """
    gamma = float(np.clip(gamma, 0.5, 2.0))
    corrected = np.power(np.clip(image, 1e-6, 1.0), 1.0 / gamma)
    return corrected.astype(np.float32)

def apply_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear contrast stretch around image mean.
    """
    alpha = float(np.clip(alpha, 0.8, 2.0))
    mean_val = float(np.mean(image))
    stretched = mean_val + alpha * (image - mean_val)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)

def apply_saturation(image: np.ndarray, gain: float) -> np.ndarray:
    """
    Saturation boost via RGB -> HSV -> RGB.
    """
    gain = float(np.clip(gain, 0.5, 2.0))
    img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * gain, 0, 255)
    bgr_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return (rgb_out / 255.0).astype(np.float32)

def isp_pipeline(image: np.ndarray,
                 gamma: float,
                 alpha: float,
                 gain: float) -> np.ndarray:
    """
    Full ISP chain: gamma -> contrast -> saturation.
    """
    image = apply_gamma(image, gamma)
    image = apply_contrast(image, alpha)
    image = apply_saturation(image, gain)
    return image

def compute_asymmetric_clipping_penalty(image: np.ndarray) -> float:
    """
    Loophole 1 Fix: Strict on white clipping (prevents deep-frying),
    relaxed on black clipping (allows natural night shadows).
    """
    total_pixels = image.size
    white_ratio = np.sum(image > 0.98) / total_pixels
    black_ratio = np.sum(image < 0.02) / total_pixels

    penalty = 0.0

    # STRICT White Clipping: Starts penalizing at 2%, maxes out at 10%
    if white_ratio > 0.02:
        penalty -= min((white_ratio - 0.02) / 0.08, 1.0)

    # RELAXED Black Clipping: Allows up to 20% natural shadows, maxes at 40%
    if black_ratio > 0.20:
        penalty -= min((black_ratio - 0.20) / 0.20, 1.0) * 0.5

    return penalty

def compute_color_variance_penalty(image: np.ndarray) -> float:
    """
    Loophole 2 Fix: Penalizes unnatural color tints by ensuring R, G, B means are balanced.
    Prevents MUSIQ from being tricked by cinematic color casts.
    """
    mean_r = np.mean(image[:,:,0])
    mean_g = np.mean(image[:,:,1])
    mean_b = np.mean(image[:,:,2])
    std_dev = float(np.std([mean_r, mean_g, mean_b]))

    # A standard deviation > 0.05 indicates a severe color tint.
    if std_dev > 0.05:
        return -1.0 * min((std_dev - 0.05) / 0.15, 1.0)
    return 0.0