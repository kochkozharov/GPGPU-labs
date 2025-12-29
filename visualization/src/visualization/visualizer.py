"""Module for 3D visualization of ray tracing scenes."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple

from .config_reader import SceneConfig, Body, Light
from .geometry import get_all_polygons
from .camera import compute_camera_trajectory


def plot_polygon(ax: Axes3D, vertices: np.ndarray, faces: List[Tuple[int, int, int]], 
                 color: np.ndarray, alpha: float = 0.7):
    """Plot a polygon (body) on 3D axes.
    
    Args:
        ax: 3D axes object
        vertices: Array of vertices, shape (n, 3)
        faces: List of face indices (v0, v1, v2)
        color: RGB color [r, g, b] in range [0, 1]
        alpha: Transparency (0-1)
    """
    triangles = []
    for face in faces:
        triangle = vertices[list(face)]
        triangles.append(triangle)
    
    collection = Poly3DCollection(triangles, alpha=alpha, facecolor=color, 
                                   edgecolor='black', linewidths=0.5)
    ax.add_collection3d(collection)


def plot_floor(ax: Axes3D, floor_points: np.ndarray, color: np.ndarray, alpha: float = 0.3):
    """Plot floor polygon.
    
    Args:
        ax: 3D axes object
        floor_points: Array of 12 coordinates (4 points * 3 coords)
        color: RGB color [r, g, b]
        alpha: Transparency
    """
    # Reshape to 4 points
    points = floor_points.reshape(4, 3)
    
    # Create floor polygon
    floor_poly = [points]
    collection = Poly3DCollection(floor_poly, alpha=alpha, facecolor=color,
                                   edgecolor='gray', linewidths=1.0)
    ax.add_collection3d(collection)


def plot_lights(ax: Axes3D, lights: List[Light], size: float = 0.3):
    """Plot light sources.
    
    Args:
        ax: 3D axes object
        lights: List of Light objects
        size: Size of light markers
    """
    for light in lights:
        ax.scatter(*light.position, c=light.color, s=size*1000, 
                  marker='*', edgecolors='yellow', linewidths=1)


def visualize_scene(config: SceneConfig, output_path: str = None):
    """Create comprehensive 3D visualization of the scene.
    
    Args:
        config: Scene configuration
        output_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3 subplots: full scene, camera trajectory, direction trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Get all polygons
    body_types = ['tetrahedron', 'hexahedron', 'icosahedron']
    polygons = get_all_polygons(config.bodies, body_types)
    
    # Plot 1: Full scene with all objects
    for vertices, faces, color in polygons:
        plot_polygon(ax1, vertices, faces, color, alpha=0.7)
    
    plot_floor(ax1, config.floor_points, config.floor_color, alpha=0.3)
    plot_lights(ax1, config.lights)
    
    # Compute camera trajectories
    cam_positions, cam_targets = compute_camera_trajectory(config.camera_params, config.num_frames)
    
    # Plot camera trajectory
    ax1.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
             'r-', linewidth=2, label='Camera path', alpha=0.6)
    ax1.scatter(cam_positions[0, 0], cam_positions[0, 1], cam_positions[0, 2],
               c='red', s=100, marker='o', label='Start')
    ax1.scatter(cam_positions[-1, 0], cam_positions[-1, 1], cam_positions[-1, 2],
               c='darkred', s=100, marker='s', label='End')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Full Scene with Camera Trajectory')
    ax1.legend()
    
    # Plot 2: Camera trajectory only
    ax2.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
             'b-', linewidth=2, alpha=0.8)
    ax2.scatter(cam_positions[0, 0], cam_positions[0, 1], cam_positions[0, 2],
               c='green', s=150, marker='o', label='Start', zorder=5)
    ax2.scatter(cam_positions[-1, 0], cam_positions[-1, 1], cam_positions[-1, 2],
               c='red', s=150, marker='s', label='End', zorder=5)
    
    # Add objects for reference
    for vertices, faces, color in polygons:
        plot_polygon(ax2, vertices, faces, color, alpha=0.2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Траектория позиции камеры')
    ax2.legend()
    
    # Plot 3: Camera target (direction) trajectory
    ax3.plot(cam_targets[:, 0], cam_targets[:, 1], cam_targets[:, 2],
             'g-', linewidth=2, alpha=0.8)
    ax3.scatter(cam_targets[0, 0], cam_targets[0, 1], cam_targets[0, 2],
               c='blue', s=150, marker='o', label='Start', zorder=5)
    ax3.scatter(cam_targets[-1, 0], cam_targets[-1, 1], cam_targets[-1, 2],
               c='darkgreen', s=150, marker='s', label='End', zorder=5)
    
    # Add objects for reference
    for vertices, faces, color in polygons:
        plot_polygon(ax3, vertices, faces, color, alpha=0.2)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Траектория направления камеры')
    ax3.legend()
    
    # Plot 4: Both trajectories together
    ax4.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
             'r-', linewidth=2, label='Camera position', alpha=0.7)
    ax4.plot(cam_targets[:, 0], cam_targets[:, 1], cam_targets[:, 2],
             'g-', linewidth=2, label='Camera target', alpha=0.7)
    
    # Draw lines connecting camera to target at several points
    num_connections = 10
    for i in range(0, len(cam_positions), len(cam_positions) // num_connections):
        ax4.plot([cam_positions[i, 0], cam_targets[i, 0]],
                 [cam_positions[i, 1], cam_targets[i, 1]],
                 [cam_positions[i, 2], cam_targets[i, 2]],
                 'k--', alpha=0.3, linewidth=0.5)
    
    # Add objects
    for vertices, faces, color in polygons:
        plot_polygon(ax4, vertices, faces, color, alpha=0.3)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Траектория позиции и направления камеры')
    ax4.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def visualize_separate_plots(config: SceneConfig, output_dir: str = None):
    """Create separate plots for each visualization.
    
    Args:
        config: Scene configuration
        output_dir: Optional directory to save figures
    """
    body_types = ['tetrahedron', 'hexahedron', 'icosahedron']
    polygons = get_all_polygons(config.bodies, body_types)
    cam_positions, cam_targets = compute_camera_trajectory(config.camera_params, config.num_frames)
    
    # Plot 1: Full scene
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    for vertices, faces, color in polygons:
        plot_polygon(ax1, vertices, faces, color, alpha=0.7)
    plot_floor(ax1, config.floor_points, config.floor_color, alpha=0.3)
    plot_lights(ax1, config.lights)
    ax1.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
             'r-', linewidth=2, label='Camera path')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Full Scene with All Polygons and Camera Trajectory')
    ax1.legend()
    
    if output_dir:
        plt.savefig(f"{output_dir}/scene_full.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot 2: Camera trajectory
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    ax2.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
             'b-', linewidth=3, label='Траектория позиции камеры')
    ax2.scatter(cam_positions[0, 0], cam_positions[0, 1], cam_positions[0, 2],
               c='green', s=200, marker='o', label='Начало')
    ax2.scatter(cam_positions[-1, 0], cam_positions[-1, 1], cam_positions[-1, 2],
               c='red', s=200, marker='s', label='Конец')
    
    for vertices, faces, color in polygons:
        plot_polygon(ax2, vertices, faces, color, alpha=0.2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Траектория позиции камеры')
    ax2.legend()
    
    if output_dir:
        plt.savefig(f"{output_dir}/camera_trajectory.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Plot 3: Target trajectory
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    ax3.plot(cam_targets[:, 0], cam_targets[:, 1], cam_targets[:, 2],
             'g-', linewidth=3, label='Траектория направления камеры')
    ax3.scatter(cam_targets[0, 0], cam_targets[0, 1], cam_targets[0, 2],
               c='blue', s=200, marker='o', label='Начало')
    ax3.scatter(cam_targets[-1, 0], cam_targets[-1, 1], cam_targets[-1, 2],
               c='darkgreen', s=200, marker='s', label='Конец')
    
    for vertices, faces, color in polygons:
        plot_polygon(ax3, vertices, faces, color, alpha=0.2)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Траектория направления камеры')
    ax3.legend()
    
    if output_dir:
        plt.savefig(f"{output_dir}/target_trajectory.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


