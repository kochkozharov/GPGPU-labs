"""Module for generating geometry of platonic solids."""

import numpy as np
from typing import List, Tuple, Any


def generate_tetrahedron(center: np.ndarray, radius: float) -> np.ndarray:
    """Generate vertices of a tetrahedron inscribed in a sphere.
    
    Args:
        center: Center point [x, y, z]
        radius: Radius of circumscribed sphere
        
    Returns:
        Array of 4 vertices, shape (4, 3)
    """
    a = radius / np.sqrt(3.0)
    
    vertices = np.array([
        [center[0] + a, center[1] + a, center[2] + a],
        [center[0] + a, center[1] - a, center[2] - a],
        [center[0] - a, center[1] + a, center[2] - a],
        [center[0] - a, center[1] - a, center[2] + a],
    ])
    
    return vertices


def generate_hexahedron(center: np.ndarray, radius: float) -> np.ndarray:
    """Generate vertices of a hexahedron (cube) inscribed in a sphere.
    
    Args:
        center: Center point [x, y, z]
        radius: Radius of circumscribed sphere
        
    Returns:
        Array of 8 vertices, shape (8, 3)
    """
    a = radius / np.sqrt(3.0)
    
    vertices = np.array([
        [center[0] - a, center[1] - a, center[2] - a],  # 0
        [center[0] + a, center[1] - a, center[2] - a],  # 1
        [center[0] + a, center[1] + a, center[2] - a],  # 2
        [center[0] - a, center[1] + a, center[2] - a],  # 3
        [center[0] - a, center[1] - a, center[2] + a],  # 4
        [center[0] + a, center[1] - a, center[2] + a],  # 5
        [center[0] + a, center[1] + a, center[2] + a],  # 6
        [center[0] - a, center[1] + a, center[2] + a],  # 7
    ])
    
    return vertices


def generate_icosahedron(center: np.ndarray, radius: float) -> np.ndarray:
    """Generate vertices of an icosahedron inscribed in a sphere.
    
    Args:
        center: Center point [x, y, z]
        radius: Radius of circumscribed sphere
        
    Returns:
        Array of 12 vertices, shape (12, 3)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    scale = radius / np.sqrt(1.0 + phi * phi)
    
    vertices = np.array([
        [center[0] - scale, center[1] + phi * scale, center[2]],
        [center[0] + scale, center[1] + phi * scale, center[2]],
        [center[0] - scale, center[1] - phi * scale, center[2]],
        [center[0] + scale, center[1] - phi * scale, center[2]],
        [center[0], center[1] - scale, center[2] + phi * scale],
        [center[0], center[1] + scale, center[2] + phi * scale],
        [center[0], center[1] - scale, center[2] - phi * scale],
        [center[0], center[1] + scale, center[2] - phi * scale],
        [center[0] + phi * scale, center[1], center[2] - scale],
        [center[0] + phi * scale, center[1], center[2] + scale],
        [center[0] - phi * scale, center[1], center[2] - scale],
        [center[0] - phi * scale, center[1], center[2] + scale],
    ])
    
    return vertices


def generate_tetrahedron_faces() -> List[Tuple[int, int, int]]:
    """Generate face indices for tetrahedron.
    
    Returns:
        List of tuples (v0, v1, v2) for each face
    """
    return [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]


def generate_hexahedron_faces() -> List[Tuple[int, int, int]]:
    """Generate face indices for hexahedron (cube).
    
    Returns:
        List of tuples (v0, v1, v2) for each triangle (12 triangles for 6 faces)
    """
    return [
        # Bottom face (z = -a)
        (0, 1, 2), (0, 2, 3),
        # Top face (z = +a)
        (4, 6, 5), (4, 7, 6),
        # Front face (y = -a)
        (0, 5, 1), (0, 4, 5),
        # Back face (y = +a)
        (2, 7, 3), (2, 6, 7),
        # Left face (x = -a)
        (0, 7, 4), (0, 3, 7),
        # Right face (x = +a)
        (1, 5, 6), (1, 6, 2),
    ]


def generate_icosahedron_faces() -> List[Tuple[int, int, int]]:
    """Generate face indices for icosahedron.
    
    Returns:
        List of tuples (v0, v1, v2) for each face (20 faces)
    """
    return [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]


def get_all_polygons(bodies: List[Any], body_types: List[str]) -> List[Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]]:
    """Get all polygons for all bodies.
    
    Args:
        bodies: List of Body objects
        body_types: List of body type names ['tetrahedron', 'hexahedron', 'icosahedron']
        
    Returns:
        List of tuples (vertices, faces, color) for each body
    """
    polygons = []
    
    for body, body_type in zip(bodies, body_types):
        if body_type == 'tetrahedron':
            vertices = generate_tetrahedron(body.center, body.radius)
            faces = generate_tetrahedron_faces()
        elif body_type == 'hexahedron':
            vertices = generate_hexahedron(body.center, body.radius)
            faces = generate_hexahedron_faces()
        elif body_type == 'icosahedron':
            vertices = generate_icosahedron(body.center, body.radius)
            faces = generate_icosahedron_faces()
        else:
            raise ValueError(f"Unknown body type: {body_type}")
        
        polygons.append((vertices, faces, body.color))
    
    return polygons

