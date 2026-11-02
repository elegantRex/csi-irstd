"""Image filename compatibility; no label or dataset rewriting."""
import os

def resolve_image_path(image_root, file_name):
    """Read the original image when COCO metadata says .jpg but stores .png."""
    path = os.path.join(image_root, file_name)
    if os.path.isfile(path):
        return path
    stem, extension = os.path.splitext(path)
    if extension.lower() in {'.jpg', '.jpeg'}:
        png_path = stem + '.png'
        if os.path.isfile(png_path):
            return png_path
    raise FileNotFoundError(f"Image is missing for annotation {file_name!r}: {path}")
