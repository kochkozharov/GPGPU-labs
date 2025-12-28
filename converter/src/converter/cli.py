"""Command-line interface for the image converter."""

import argparse
import sys
from pathlib import Path

from converter.converter import binary_to_png, png_to_binary, frames_to_video, video_from_directory


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert between custom binary image format and PNG/MP4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert binary to PNG
  image-converter to-png input.data output.png
  
  # Convert PNG to binary
  image-converter to-binary input.png output.data
  
  # Ignore alpha channel (set to maximum opacity)
  image-converter to-png input.data output.png --ignore-alpha
  image-converter to-binary input.png output.data --ignore-alpha
  
  # Create video from frame sequence (pattern with %d)
  image-converter to-video "frames/frame_%d.data" output.mp4 --fps 30
  
  # Create video from all .data files in a directory
  image-converter to-video-dir ./frames output.mp4 --fps 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Conversion direction', required=True)
    
    # to-png command
    to_png_parser = subparsers.add_parser(
        'to-png',
        help='Convert binary image to PNG'
    )
    to_png_parser.add_argument(
        'input',
        type=Path,
        help='Input binary image file (.data)'
    )
    to_png_parser.add_argument(
        'output',
        type=Path,
        help='Output PNG file'
    )
    to_png_parser.add_argument(
        '--ignore-alpha',
        action='store_true',
        help='Set alpha channel to maximum (fully opaque) for all pixels'
    )
    
    # to-binary command
    to_binary_parser = subparsers.add_parser(
        'to-binary',
        help='Convert PNG image to binary format'
    )
    to_binary_parser.add_argument(
        'input',
        type=Path,
        help='Input PNG image file'
    )
    to_binary_parser.add_argument(
        'output',
        type=Path,
        help='Output binary image file (.data)'
    )
    to_binary_parser.add_argument(
        '--ignore-alpha',
        action='store_true',
        help='Set alpha channel to maximum (fully opaque) for all pixels'
    )
    
    # to-video command (from pattern)
    to_video_parser = subparsers.add_parser(
        'to-video',
        help='Convert sequence of binary frames to MP4 video'
    )
    to_video_parser.add_argument(
        'pattern',
        type=str,
        help='Pattern for frame files with %%d placeholder (e.g., "frames/frame_%%d.data")'
    )
    to_video_parser.add_argument(
        'output',
        type=Path,
        help='Output MP4 video file'
    )
    to_video_parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    to_video_parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='First frame number (default: 0)'
    )
    to_video_parser.add_argument(
        '--end-frame',
        type=int,
        default=None,
        help='Last frame number (exclusive, default: auto-detect)'
    )
    to_video_parser.add_argument(
        '--ignore-alpha',
        action='store_true',
        default=True,
        help='Set alpha channel to maximum (default: True)'
    )
    to_video_parser.add_argument(
        '--no-ignore-alpha',
        action='store_false',
        dest='ignore_alpha',
        help='Keep original alpha channel values'
    )
    to_video_parser.add_argument(
        '--codec',
        type=str,
        default='libx264',
        help='Video codec (default: libx264)'
    )
    to_video_parser.add_argument(
        '--crf',
        type=int,
        default=23,
        help='Constant Rate Factor for quality, 0-51 (default: 23, lower = better)'
    )
    
    # to-video-dir command (from directory)
    to_video_dir_parser = subparsers.add_parser(
        'to-video-dir',
        help='Convert all binary frames in a directory to MP4 video'
    )
    to_video_dir_parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing binary image files'
    )
    to_video_dir_parser.add_argument(
        'output',
        type=Path,
        help='Output MP4 video file'
    )
    to_video_dir_parser.add_argument(
        '--pattern',
        type=str,
        default='*.data',
        help='Glob pattern for frame files (default: *.data)'
    )
    to_video_dir_parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    to_video_dir_parser.add_argument(
        '--ignore-alpha',
        action='store_true',
        default=True,
        help='Set alpha channel to maximum (default: True)'
    )
    to_video_dir_parser.add_argument(
        '--no-ignore-alpha',
        action='store_false',
        dest='ignore_alpha',
        help='Keep original alpha channel values'
    )
    to_video_dir_parser.add_argument(
        '--codec',
        type=str,
        default='libx264',
        help='Video codec (default: libx264)'
    )
    to_video_dir_parser.add_argument(
        '--crf',
        type=int,
        default=23,
        help='Constant Rate Factor for quality, 0-51 (default: 23, lower = better)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'to-png':
            # Validate input file exists
            if not args.input.exists():
                print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
                sys.exit(1)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            binary_to_png(args.input, args.output, ignore_alpha=args.ignore_alpha)
            print(f"Successfully converted '{args.input}' to '{args.output}'")
            
        elif args.command == 'to-binary':
            if not args.input.exists():
                print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
                sys.exit(1)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            png_to_binary(args.input, args.output, ignore_alpha=args.ignore_alpha)
            print(f"Successfully converted '{args.input}' to '{args.output}'")
            
        elif args.command == 'to-video':
            frames_to_video(
                frame_pattern=args.pattern,
                output_path=args.output,
                fps=args.fps,
                ignore_alpha=args.ignore_alpha,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                codec=args.codec,
                crf=args.crf
            )
            
        elif args.command == 'to-video-dir':
            if not args.input_dir.exists():
                print(f"Error: Directory '{args.input_dir}' does not exist", file=sys.stderr)
                sys.exit(1)
            video_from_directory(
                input_dir=args.input_dir,
                output_path=args.output,
                pattern=args.pattern,
                fps=args.fps,
                ignore_alpha=args.ignore_alpha,
                codec=args.codec,
                crf=args.crf
            )
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
