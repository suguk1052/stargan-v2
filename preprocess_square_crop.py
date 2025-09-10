import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def crop_square(img: Image.Image, bottom_right: bool = False) -> Image.Image:
    """Crop a square region from an image.

    By default the top-left square region is returned. If ``bottom_right`` is
    True, the bottom-most or right-most square region is used depending on the
    image orientation.
    """
    width, height = img.size
    size = min(width, height)
    if bottom_right:
        if width >= height:
            left, upper = width - size, 0  # crop from right
        else:
            left, upper = 0, height - size  # crop from bottom
        return img.crop((left, upper, left + size, upper + size))
    return img.crop((0, 0, size, size))


def process_file(src: Path, dst: Path, bottom_right: bool = False) -> None:
    with Image.open(src) as img:
        cropped = crop_square(img, bottom_right=bottom_right)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dst)


def process_path(input_path: Path, output_path: Path, bottom_right: bool = False) -> None:
    if input_path.is_file():
        process_file(input_path, output_path, bottom_right)
    else:
        for file in input_path.glob('**/*'):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                rel = file.relative_to(input_path)
                process_file(file, output_path / rel, bottom_right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Crop the top-left square region from images and save them to an output path. '
            'Use --bottom-right to crop the bottom or right square region instead.'
        )
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        type=str,
        help='Path to an image file or a directory containing images.',
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        type=str,
        help='Path to save the cropped images.',
    )
    parser.add_argument(
        '--bottom-right',
        action='store_true',
        help='Crop the bottom or right square region instead of the default top or left.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_path(Path(args.input), Path(args.output), args.bottom_right)


if __name__ == '__main__':
    main()
