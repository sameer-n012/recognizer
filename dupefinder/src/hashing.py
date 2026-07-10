"""Hand-rolled perceptual hashing (aHash/dHash/pHash) built on opencv + numpy only —
deliberately avoids adding an `imagehash` dependency. Hashes are 64-bit (hash_size=8)
and stored/compared as hex strings so they round-trip cleanly through parquet.
"""

import cv2
import numpy as np


def _to_gray_resized(img: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if resized.ndim == 3:
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return format(value, "016x")


def average_hash(img: np.ndarray, hash_size: int = 8) -> str:
    gray = _to_gray_resized(img, hash_size).astype(np.float32)
    mean = gray.mean()
    return _bits_to_hex((gray > mean).flatten())


def difference_hash(img: np.ndarray, hash_size: int = 8) -> str:
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = resized[:, 1:] > resized[:, :-1]
    return _bits_to_hex(diff.flatten())


def perceptual_hash(img: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    gray = _to_gray_resized(img, size).astype(np.float32)
    dct = cv2.dct(gray)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    return _bits_to_hex((dct_low > median).flatten())


def compute_hashes(img: np.ndarray, hash_size: int = 8) -> dict[str, str]:
    return {
        "ahash": average_hash(img, hash_size),
        "dhash": difference_hash(img, hash_size),
        "phash": perceptual_hash(img, hash_size),
    }


def hamming_distance(hex_a: str, hex_b: str) -> int:
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")
