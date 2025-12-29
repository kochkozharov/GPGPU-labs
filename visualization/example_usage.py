#!/usr/bin/env python3
"""Example usage of the visualization package."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualization.config_reader import read_config
from visualization.visualizer import visualize_scene, visualize_separate_plots

if __name__ == "__main__":
    # Example: visualize config1
    config_path = Path(__file__).parent.parent / "cp" / "input" / "config1"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please provide a config file path as argument")
        if len(sys.argv) > 1:
            config_path = Path(sys.argv[1])
        else:
            sys.exit(1)
    
    print(f"Loading config from {config_path}")
    config = read_config(str(config_path))
    
    print(f"Scene: {len(config.bodies)} bodies, {len(config.lights)} lights")
    print(f"Frames: {config.num_frames}")
    print(f"Resolution: {config.width}x{config.height}")
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_scene(config, output_path="scene_visualization.png")
    
    print("\nDone! Check scene_visualization.png")


