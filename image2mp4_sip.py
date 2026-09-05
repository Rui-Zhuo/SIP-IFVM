import os
import re
import cv2
from PIL import Image


def extract_time_from_filename(filename):
    """
    Extract simulation time [hour] from filenames such as:

        Br_slices_time.82.00h.r.0.png
        Br_slices_time.102.00h.r.0.png

    Returns
    -------
    float or None
        Simulation time in hours.
        Returns None if the filename does not contain a parsable time.
    """
    match = re.search(
        r"time\.([0-9]+(?:\.[0-9]+)?)h",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return float(match.group(1))


def image_sort_key(filename):
    """
    Sort image files by NUMERIC simulation time.

    Parsed files are sorted first by time.
    Files without a parsable time are placed afterward and sorted
    alphabetically.
    """
    simulation_time = extract_time_from_filename(
        filename
    )

    if simulation_time is None:
        return (
            1,
            float("inf"),
            filename.lower(),
        )

    return (
        0,
        simulation_time,
        filename.lower(),
    )


def image2video(
    image_dir,
    video_path,
    fps=3,
):
    """
    Merge images into an MP4 video.

    Parameters
    ----------
    image_dir : str
        Directory containing source images.

    video_path : str
        Output MP4 path.

    fps : int or float
        Frames per second.
    """

    # --------------------------------------------------------------
    # Step 1: Get valid image filenames
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Step 2: Sort by NUMERIC simulation time
    #
    # Example:
    #   Br_slices_time.82.00h.r.0.png
    #   Br_slices_time.102.00h.r.0.png
    #
    # Correct order:
    #   82.00 h -> 102.00 h
    # --------------------------------------------------------------
    image_names = sorted(
        image_names,
        key=image_sort_key,
    )

    print("Image order:")

    for i, image_name in enumerate(
        image_names,
        start=1,
    ):
        simulation_time = extract_time_from_filename(
            image_name
        )

        if simulation_time is None:
            time_text = "time = unknown"
        else:
            time_text = (
                f"time = {simulation_time:.2f} h"
            )

        print(
            f"{i:4d}: {image_name}    ({time_text})"
        )

    # --------------------------------------------------------------
    # Step 3: Read first frame and initialize VideoWriter
    # --------------------------------------------------------------
    first_image_path = os.path.join(
        image_dir,
        image_names[0],
    )

    first_image = Image.open(
        first_image_path
    )

    frame_size = first_image.size

    fourcc = cv2.VideoWriter_fourcc(
        "m",
        "p",
        "4",
        "v",
    )

    video_writer = cv2.VideoWriter(
        video_path,
        fourcc,
        fps,
        frame_size,
    )

    if not video_writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for: {video_path}"
        )

    # --------------------------------------------------------------
    # Step 4: Write frames
    # --------------------------------------------------------------
    written_frames = 0

    for idx, image_name in enumerate(
        image_names,
        start=1,
    ):
        image_path = os.path.join(
            image_dir,
            image_name,
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            print(
                f"Warning: failed to read {image_name}, skipped."
            )
            continue

        height, width = image.shape[:2]

        if (
            width != frame_size[0]
            or height != frame_size[1]
        ):
            raise ValueError(
                f"Image size mismatch for {image_name}: "
                f"{width}x{height}, "
                f"expected {frame_size[0]}x{frame_size[1]}"
            )

        video_writer.write(
            image
        )

        written_frames += 1

        print(
            f"Writing {image_name} "
            f"({idx}/{len(image_names)}) Done!"
        )

    # --------------------------------------------------------------
    # Step 5: Release video
    # --------------------------------------------------------------
    video_writer.release()

    if written_frames == 0:
        raise RuntimeError(
            "No valid frames were written."
        )

    print(
        f"Converting completed! Video saved to: {video_path}"
    )

    print(
        f"Video info: {written_frames} frames, "
        f"{fps} FPS"
    )


root_dir = "F:/Data/SIP-IFVM/slices/82d1to132/"
sub_dir = "vr"

image_dir = os.path.join(
    root_dir,
    sub_dir + "/",
)

video_path = os.path.join(
    root_dir,
    sub_dir + ".mp4",
)

image2video(
    image_dir=image_dir,
    video_path=video_path,
    fps=8,
)
