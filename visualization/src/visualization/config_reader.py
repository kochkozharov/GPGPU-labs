"""Module for reading ray tracing configuration files."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CameraParams:
    """Camera animation parameters in cylindrical coordinates."""
    # Position parameters
    r_c0: float
    z_c0: float
    phi_c0: float
    A_c_r: float
    A_c_z: float
    omega_c_r: float
    omega_c_z: float
    omega_c_phi: float
    p_c_r: float
    p_c_z: float
    
    # Direction (target) parameters
    r_n0: float
    z_n0: float
    phi_n0: float
    A_n_r: float
    A_n_z: float
    omega_n_r: float
    omega_n_z: float
    omega_n_phi: float
    p_n_r: float
    p_n_z: float


@dataclass
class Body:
    """Geometric body definition."""
    center: np.ndarray  # [x, y, z]
    color: np.ndarray   # [r, g, b]
    radius: float
    reflection: float
    transparency: float
    edge_lights: int


@dataclass
class Light:
    """Light source definition."""
    position: np.ndarray  # [x, y, z]
    color: np.ndarray     # [r, g, b]


@dataclass
class SceneConfig:
    """Complete scene configuration."""
    num_frames: int
    output_path: str
    width: int
    height: int
    fov_deg: float
    camera_params: CameraParams
    bodies: List[Body]
    floor_points: np.ndarray  # 4 points, 12 coordinates
    floor_texture: str
    floor_color: np.ndarray
    floor_reflection: float
    lights: List[Light]
    max_depth: int
    ssaa_sqrt: int


def read_config(config_path: str) -> SceneConfig:
    """Read scene configuration from file."""
    with open(config_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Frame count
    num_frames = int(lines[0])
    
    # Output path
    output_path = lines[1]
    
    # Resolution and FOV
    width, height, fov_deg = map(float, lines[2].split())
    width, height = int(width), int(height)
    
    # Camera position parameters
    cam_pos_params = list(map(float, lines[3].split()))
    if len(cam_pos_params) != 10:
        raise ValueError("Expected 10 camera position parameters")
    
    # Camera direction parameters
    cam_dir_params = list(map(float, lines[4].split()))
    if len(cam_dir_params) != 10:
        raise ValueError("Expected 10 camera direction parameters")
    
    camera_params = CameraParams(
        r_c0=cam_pos_params[0],
        z_c0=cam_pos_params[1],
        phi_c0=cam_pos_params[2],
        A_c_r=cam_pos_params[3],
        A_c_z=cam_pos_params[4],
        omega_c_r=cam_pos_params[5],
        omega_c_z=cam_pos_params[6],
        omega_c_phi=cam_pos_params[7],
        p_c_r=cam_pos_params[8],
        p_c_z=cam_pos_params[9],
        r_n0=cam_dir_params[0],
        z_n0=cam_dir_params[1],
        phi_n0=cam_dir_params[2],
        A_n_r=cam_dir_params[3],
        A_n_z=cam_dir_params[4],
        omega_n_r=cam_dir_params[5],
        omega_n_z=cam_dir_params[6],
        omega_n_phi=cam_dir_params[7],
        p_n_r=cam_dir_params[8],
        p_n_z=cam_dir_params[9],
    )
    
    # Bodies (3: tetrahedron, hexahedron, icosahedron)
    bodies = []
    for i in range(3):
        body_line = lines[5 + i].split()
        body_params = list(map(float, body_line[:-1]))  # All except last are floats
        edge_lights = int(float(body_line[-1]))  # Last is int (but may be written as float)
        
        center = np.array([body_params[0], body_params[1], body_params[2]])
        color = np.array([body_params[3], body_params[4], body_params[5]])
        radius = body_params[6]
        reflection = body_params[7]
        transparency = body_params[8]
        
        bodies.append(Body(
            center=center,
            color=color,
            radius=radius,
            reflection=reflection,
            transparency=transparency,
            edge_lights=edge_lights
        ))
    
    # Floor
    floor_data = lines[8].split()
    floor_points = np.array(list(map(float, floor_data[:12])))
    floor_texture = floor_data[12]
    floor_color = np.array(list(map(float, floor_data[13:16])))
    floor_reflection = float(floor_data[16])
    
    # Lights
    num_lights = int(lines[9])
    lights = []
    for i in range(num_lights):
        light_params = list(map(float, lines[10 + i].split()))
        position = np.array([light_params[0], light_params[1], light_params[2]])
        color = np.array([light_params[3], light_params[4], light_params[5]])
        lights.append(Light(position=position, color=color))
    
    # Depth and SSAA
    depth_ssaa_line = lines[9 + num_lights].split()
    max_depth = int(float(depth_ssaa_line[0]))  # May be written as float
    ssaa_sqrt = int(float(depth_ssaa_line[1]))  # May be written as float
    
    return SceneConfig(
        num_frames=num_frames,
        output_path=output_path,
        width=width,
        height=height,
        fov_deg=fov_deg,
        camera_params=camera_params,
        bodies=bodies,
        floor_points=floor_points,
        floor_texture=floor_texture,
        floor_color=floor_color,
        floor_reflection=floor_reflection,
        lights=lights,
        max_depth=max_depth,
        ssaa_sqrt=ssaa_sqrt
    )

