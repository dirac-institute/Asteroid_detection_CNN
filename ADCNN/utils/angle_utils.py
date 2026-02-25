"""
Angle unit conversion utilities.

Convention: CSV catalog stores 'beta' in DEGREES.
All geometric calculations use RADIANS internally.

Variable naming convention:
  - *_deg : angles in degrees
  - *_rad : angles in radians
  - beta (no suffix in CSV) : degrees (legacy)

Usage:
    from ADCNN.utils.angle_utils import deg2rad, ensure_radians

    # When reading from CSV:
    beta_deg = float(row['beta'])  # CSV has degrees
    beta_rad = deg2rad(beta_deg)

    # Or auto-detect with safety:
    beta_rad = ensure_radians(beta_value, assume_degrees=True)
"""

import numpy as np
from typing import Union


def deg2rad(angle_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angle from degrees to radians.

    Args:
        angle_deg: Angle in degrees (scalar or array)

    Returns:
        Angle in radians (same type as input)
    """
    return np.deg2rad(angle_deg)


def rad2deg(angle_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angle from radians to degrees.

    Args:
        angle_rad: Angle in radians (scalar or array)

    Returns:
        Angle in degrees (same type as input)
    """
    return np.rad2deg(angle_rad)


def ensure_radians(angle: Union[float, np.ndarray], assume_degrees: bool = True) -> Union[float, np.ndarray]:
    """
    Ensure angle is in radians, with heuristic detection for common errors.

    This is a safety function that attempts to detect if an angle is in degrees
    when it should be in radians, based on magnitude.

    Args:
        angle: Angle value (scalar or array)
        assume_degrees: If True and angle is large (|angle| > 2*pi),
                       assume it's in degrees and convert

    Returns:
        Angle in radians (same type as input)

    Warning:
        This heuristic fails for angles > 360 degrees. Use explicit conversion instead.
    """
    if assume_degrees:
        # If magnitude suggests degrees (> 2*pi ~= 6.28), convert
        threshold = 2 * np.pi
        if isinstance(angle, np.ndarray):
            mask = np.abs(angle) > threshold
            result = angle.copy()
            result[mask] = np.deg2rad(result[mask])
            return result
        else:
            if abs(float(angle)) > threshold:
                return np.deg2rad(angle)
    return angle


def normalize_angle_rad(angle_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize angle to [-pi, pi] range.

    Args:
        angle_rad: Angle in radians (scalar or array)

    Returns:
        Normalized angle in radians
    """
    if isinstance(angle_rad, np.ndarray):
        return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
    else:
        return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def normalize_angle_deg(angle_deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Normalize angle to [-180, 180] range.

    Args:
        angle_deg: Angle in degrees (scalar or array)

    Returns:
        Normalized angle in degrees
    """
    angle_rad = deg2rad(angle_deg)
    normalized_rad = normalize_angle_rad(angle_rad)
    return rad2deg(normalized_rad)

