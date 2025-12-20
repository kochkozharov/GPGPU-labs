"""Core conversion functions for binary image format and PNG."""

import struct
from pathlib import Path
from typing import Tuple

from PIL import Image
import numpy as np


def read_binary_image(filepath: Path) -> Tuple[int, int, np.ndarray]:
    """
    Read a binary image file in the custom format.
    
    Format:
    - First 4 bytes: width (int, little-endian)
    - Next 4 bytes: height (int, little-endian)
    - Remaining bytes: pixel data row by row
    - Each pixel is 4 bytes: RGBA format (R, G, B, A)
    
    Args:
        filepath: Path to the binary image file
        
    Returns:
        Tuple of (width, height, image_array) where image_array is shape (height, width, 4)
        with RGBA channels
    """
    with open(filepath, 'rb') as f:
        # Read width and height (4 bytes each, little-endian int)
        width_bytes = f.read(4)
        height_bytes = f.read(4)
        
        if len(width_bytes) != 4 or len(height_bytes) != 4:
            raise ValueError("File too short: missing width/height")
        
        width = struct.unpack('<i', width_bytes)[0]
        height = struct.unpack('<i', height_bytes)[0]
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        
        # Read pixel data
        expected_bytes = width * height * 4
        pixel_data = f.read(expected_bytes)
        
        if len(pixel_data) != expected_bytes:
            raise ValueError(f"File too short: expected {expected_bytes} bytes of pixel data, got {len(pixel_data)}")
        
        # Convert to numpy array
        # Each pixel is 4 bytes: R, G, B, A
        pixels = np.frombuffer(pixel_data, dtype=np.uint8)
        image_array = pixels.reshape((height, width, 4))
        
        return width, height, image_array


def write_binary_image(filepath: Path, image_array: np.ndarray) -> None:
    """
    Write an image array to a binary file in the custom format.
    
    Args:
        filepath: Path where to write the binary image file
        image_array: Image array of shape (height, width, 4) with RGBA channels
    """
    height, width = image_array.shape[:2]
    
    with open(filepath, 'wb') as f:
        # Write width and height (4 bytes each, little-endian int)
        f.write(struct.pack('<i', width))
        f.write(struct.pack('<i', height))
        
        # Write pixel data row by row
        # Ensure the array is contiguous and in the right format
        if image_array.shape[2] != 4:
            raise ValueError(f"Image must have 4 channels (RGBA), got {image_array.shape[2]}")
        
        # Flatten to row-major order and write
        pixel_data = image_array.astype(np.uint8).tobytes()
        f.write(pixel_data)


def binary_to_png(binary_path: Path, png_path: Path, ignore_alpha: bool = False) -> None:
    """
    Convert a binary image file to PNG.
    
    Args:
        binary_path: Path to the input binary image file
        png_path: Path where to save the PNG file
        ignore_alpha: If True, set alpha channel to maximum (255) for all pixels
    """
    width, height, image_array = read_binary_image(binary_path)
    
    # Make a writable copy if we need to modify alpha channel
    if ignore_alpha:
        image_array = image_array.copy()
        # Set alpha channel to maximum (fully opaque)
        image_array[:, :, 3] = 255
    
    # Create PIL Image from array
    # PIL expects RGB or RGBA mode
    image = Image.fromarray(image_array, 'RGBA')
    
    # Save as PNG
    image.save(png_path, 'PNG')


def png_to_binary(png_path: Path, binary_path: Path, ignore_alpha: bool = False) -> None:
    """
    Convert a PNG image file to binary format.
    
    Args:
        png_path: Path to the input PNG image file
        binary_path: Path where to save the binary file
        ignore_alpha: If True, set alpha channel to maximum (255) for all pixels
    """
    # Load PNG image
    image = Image.open(png_path)
    
    # Convert to RGBA if not already
    if image.mode != 'RGBA':
        if image.mode == 'RGB':
            # Add alpha channel
            image = image.convert('RGBA')
        else:
            # Convert through RGB first
            image = image.convert('RGB').convert('RGBA')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    if ignore_alpha:
        # image_array = image_array.copy()
        # Set alpha channel to maximum (fully opaque)
        image_array[:, :, 3] = 255
    
    # Write to binary format
    write_binary_image(binary_path, image_array)

