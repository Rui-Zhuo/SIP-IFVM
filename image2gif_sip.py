import os
import re
from PIL import Image


def extract_time_from_filename(filename):
    """
    Extract simulation time [hour] from filenames such as:
        Br_slices_time.82.00h.r.0.png
        Br_slices_time.102.00h.r.0.png

    Returns None if the time cannot be parsed.
    """
    match = re.search(
        r"time\\.([0-9]+(?:\\.[0-9]+)?)h",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return float(match.group(1))


def image_sort_key(filename):
    """
    Sort by numeric simulation time instead of string order.

    Parsed files come first.
    Unparsed files are placed afterward and sorted alphabetically.
    """
    simulation_time = extract_time_from_filename(filename)

    if simulation_time is None:
        return (1, float("inf"), filename.lower())

    return (0, simulation_time, filename.lower())


def image2gif(image_dir, gif_path, fps=3, loop=0):
    """
    Merge images into a GIF animation.
    """
    valid_ext = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
    )

    image_names = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(valid_ext)
    ]

    if not image_names:
        raise ValueError(
            f"No valid images found in directory: {image_dir}"
        )

    # Sort by NUMERIC simulation time.
    image_names = sorted(
        image_names,
        key=image_sort_key,
    )

    print("Image order:")
    for i, name in enumerate(image_names, start=1):
        t = extract_time_from_filename(name)

        if t is None:
            time_text = "time = unknown"
        else:
            time_text = f"time = {t:.2f} h"

        print(
            f"{i:4d}: {name}    ({time_text})"
        )

    frame_duration = int(1000 / fps)

    frames = []

    for idx, image_name in enumerate(image_names):
        image_path = os.path.join(
            image_dir,
            image_name,
        )

        try:
            img = Image.open(
                image_path
            ).convert("RGB")

            frames.append(img)

            print(
                f"Loading {image_name} "
                f"({idx + 1}/{len(image_names)}) Done!"
            )

        except Exception as e:
            print(
                f"Warning: Failed to load {image_name}, "
                f"error: {e}"
            )

    if not frames:
        raise RuntimeError(
            "No valid images were loaded, cannot generate GIF"
        )

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop,
        optimize=True,
    )

    print(
        f"Converting completed! GIF saved to: {gif_path}"
    )

    print(
        f"GIF info: {len(frames)} frames, "
        f"{fps} FPS, loop: {loop} times"
    )


root_dir = "F:/Data/SIP-IFVM/slices/82d1to132/"
sub_dir = "vr"

image_dir = os.path.join(
    root_dir,
    sub_dir + "/",
)

gif_path = os.path.join(
    root_dir,
    sub_dir + ".gif",
)

image2gif(
    image_dir=image_dir,
    gif_path=gif_path,
    fps=8,
    loop=0,
)
