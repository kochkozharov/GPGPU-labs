"""Command-line interface for the image converter."""

import argparse
import sys
from pathlib import Path

from converter.converter import binary_to_png, png_to_binary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert between custom binary image format and PNG',
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
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.command == 'to-png':
            binary_to_png(args.input, args.output, ignore_alpha=args.ignore_alpha)
            print(f"Successfully converted '{args.input}' to '{args.output}'")
        elif args.command == 'to-binary':
            png_to_binary(args.input, args.output, ignore_alpha=args.ignore_alpha)
            print(f"Successfully converted '{args.input}' to '{args.output}'")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

