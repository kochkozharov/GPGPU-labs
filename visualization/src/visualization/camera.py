"""Module for computing camera trajectories."""

import numpy as np
from typing import Tuple
from .config_reader import CameraParams


def compute_camera_position(params: CameraParams, t: float) -> np.ndarray:
    """Compute camera position at time t in cylindrical coordinates.
    
    Args:
        params: Camera parameters
        t: Time parameter (0 to 2*pi for full cycle)
        
    Returns:
        Camera position [x, y, z] in Cartesian coordinates
    """
    r_c = params.r_c0 + params.A_c_r * np.sin(params.omega_c_r * t + params.p_c_r)
    z_c = params.z_c0 + params.A_c_z * np.sin(params.omega_c_z * t + params.p_c_z)
    phi_c = params.phi_c0 + params.omega_c_phi * t
    
    x = r_c * np.cos(phi_c)
    y = r_c * np.sin(phi_c)
    z = z_c
    
    return np.array([x, y, z])


def compute_camera_target(params: CameraParams, t: float) -> np.ndarray:
    """Compute camera target (direction point) at time t.
    
    Args:
        params: Camera parameters
        t: Time parameter (0 to 2*pi for full cycle)
        
    Returns:
        Target position [x, y, z] in Cartesian coordinates
    """
    r_n = params.r_n0 + params.A_n_r * np.sin(params.omega_n_r * t + params.p_n_r)
    z_n = params.z_n0 + params.A_n_z * np.sin(params.omega_n_z * t + params.p_n_z)
    phi_n = params.phi_n0 + params.omega_n_phi * t
    
    x = r_n * np.cos(phi_n)
    y = r_n * np.sin(phi_n)
    z = z_n
    
    return np.array([x, y, z])


def compute_camera_trajectory(params: CameraParams, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute full camera trajectory over all frames.
    
    Args:
        params: Camera parameters
        num_frames: Number of frames in animation
        
    Returns:
        Tuple of (camera_positions, target_positions)
        Each is an array of shape (num_frames, 3)
    """
    positions = []
    targets = []
    
    for frame in range(num_frames):
        t = (frame / num_frames) * 2.0 * np.pi
        pos = compute_camera_position(params, t)
        target = compute_camera_target(params, t)
        positions.append(pos)
        targets.append(target)
    
    return np.array(positions), np.array(targets)


