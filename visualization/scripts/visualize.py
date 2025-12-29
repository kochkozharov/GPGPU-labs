#!/usr/bin/env python3
"""Main script for visualizing ray tracing scenes."""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from visualization.config_reader import read_config
from visualization.visualizer import visualize_scene, visualize_separate_plots


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <config_file> [--separate] [--output <path>]")
        print("\nExample:")
        print("  python visualize.py ../cp/input/config1")
        print("  python visualize.py ../cp/input/config1 --separate --output ./output")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Parse options
    separate = "--separate" in sys.argv
    output_path = None
    output_dir = None
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]
            if separate:
                output_dir = output_path
                os.makedirs(output_dir, exist_ok=True)
    
    # Read configuration
    print(f"Reading configuration from {config_path}...")
    try:
        config = read_config(config_path)
        print(f"Loaded scene with {len(config.bodies)} bodies, {len(config.lights)} lights, {config.num_frames} frames")
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)
    
    # Visualize
    if separate:
        print("Creating separate plots...")
        visualize_separate_plots(config, output_dir)
    else:
        print("Creating combined visualization...")
        visualize_scene(config, output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()

