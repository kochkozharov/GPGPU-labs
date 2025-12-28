"""Core conversion functions for binary image format and PNG."""

import struct
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

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


def frames_to_video(
    frame_pattern: str,
    output_path: Path,
    fps: int = 30,
    ignore_alpha: bool = True,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    codec: str = 'libx264',
    crf: int = 23
) -> None:
    """
    Convert a sequence of binary image frames to an MP4 video.
    
    Args:
        frame_pattern: Pattern for frame files with %d placeholder (e.g., "frame_%d.data")
        output_path: Path where to save the MP4 video
        fps: Frames per second for the output video
        ignore_alpha: If True, set alpha channel to maximum (fully opaque) for all frames
        start_frame: First frame number to include
        end_frame: Last frame number to include (exclusive). If None, auto-detect.
        codec: Video codec to use (default: libx264)
        crf: Constant Rate Factor for quality (0-51, lower = better quality, default: 23)
    """
    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("ffmpeg is required but not found in PATH. Please install ffmpeg.")
    
    # Create a temporary directory for PNG frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Detect frames if end_frame not specified
        if end_frame is None:
            frame_num = start_frame
            while True:
                frame_file = Path(frame_pattern % frame_num)
                if not frame_file.exists():
                    break
                frame_num += 1
            end_frame = frame_num
        
        if end_frame <= start_frame:
            raise ValueError(f"No frames found matching pattern: {frame_pattern}")
        
        print(f"Converting {end_frame - start_frame} frames to PNG...")
        
        # Convert each binary frame to PNG
        for i, frame_num in enumerate(range(start_frame, end_frame)):
            binary_path = Path(frame_pattern % frame_num)
            if not binary_path.exists():
                raise FileNotFoundError(f"Frame not found: {binary_path}")
            
            # Use sequential numbering for ffmpeg
            png_path = temp_path / f"frame_{i:06d}.png"
            binary_to_png(binary_path, png_path, ignore_alpha=ignore_alpha)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == end_frame - start_frame - 1:
                print(f"  Converted {i + 1}/{end_frame - start_frame} frames")
        
        print(f"Encoding video with ffmpeg...")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
            '-i', str(temp_path / 'frame_%06d.png'),
            '-c:v', codec,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',  # Compatibility with most players
            '-movflags', '+faststart',  # Enable streaming
            str(output_path)
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        print(f"Video saved to: {output_path}")


def video_from_directory(
    input_dir: Path,
    output_path: Path,
    pattern: str = "*.data",
    fps: int = 30,
    ignore_alpha: bool = True,
    codec: str = 'libx264',
    crf: int = 23
) -> None:
    """
    Convert all binary image files in a directory to an MP4 video.
    
    Files are sorted alphabetically before combining.
    
    Args:
        input_dir: Directory containing binary image files
        output_path: Path where to save the MP4 video
        pattern: Glob pattern for frame files (default: "*.data")
        fps: Frames per second for the output video
        ignore_alpha: If True, set alpha channel to maximum for all frames
        codec: Video codec to use
        crf: Constant Rate Factor for quality
    """
    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("ffmpeg is required but not found in PATH. Please install ffmpeg.")
    
    # Find all matching files
    import glob
    frame_files = sorted(glob.glob(str(input_dir / pattern)))
    
    if not frame_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {input_dir}")
    
    # Create a temporary directory for PNG frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"Converting {len(frame_files)} frames to PNG...")
        
        # Convert each binary frame to PNG
        for i, binary_file in enumerate(frame_files):
            binary_path = Path(binary_file)
            png_path = temp_path / f"frame_{i:06d}.png"
            binary_to_png(binary_path, png_path, ignore_alpha=ignore_alpha)
            
            if (i + 1) % 10 == 0 or i == len(frame_files) - 1:
                print(f"  Converted {i + 1}/{len(frame_files)} frames")
        
        print(f"Encoding video with ffmpeg...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', str(temp_path / 'frame_%06d.png'),
            '-c:v', codec,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(output_path)
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        print(f"Video saved to: {output_path}")
