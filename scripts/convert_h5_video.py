import argparse
from pathlib import Path
import random

import cv2
import h5py
import numpy as np


def create_video_from_h5(h5_path: str, output_dir: str, fps: int = 10, video_format: str = "mp4", max_trajs: int = 5):
    """
    Convert images from H5 file to videos for visualization.

    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the videos
        fps: Frames per second for the output videos
        video_format: Video format (mp4, avi)
        max_trajs: Maximum number of trajectories to process (default: 5)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        print(f"Found {len(traj_keys)} trajectories in the H5 file")

        # Randomly select max_trajs trajectories
        if len(traj_keys) > max_trajs:
            selected_trajs = random.sample(traj_keys, max_trajs)
            print(f"Randomly selected {max_trajs} trajectories: {selected_trajs}")
        else:
            selected_trajs = traj_keys
            print(f"Using all {len(traj_keys)} available trajectories")

        # Get all camera types available
        if selected_trajs:
            sample_traj = selected_trajs[0]
            try:
                sensor_data_path = f"{sample_traj}/obs/sensor_data"
                if sensor_data_path in f:
                    camera_types = [k for k in f[sensor_data_path].keys() if "camera" in k]
                    print(f"Available cameras: {camera_types}")
                else:
                    print("No sensor data found in the H5 file")
                    return
            except Exception as e:
                print(f"Error accessing sensor data: {e}")
                return

        # Process each camera type
        for camera_type in camera_types:
            print(f"\nProcessing {camera_type}...")

            # Collect all images for this camera across selected trajectories
            all_images = []
            valid_trajs = []

            for traj_key in selected_trajs:
                try:
                    image_path = f"{traj_key}/obs/sensor_data/{camera_type}/rgb"
                    if image_path in f:
                        images = f[image_path][:]  # (T, H, W, 3)
                        if len(images) > 0:
                            all_images.append(images)
                            valid_trajs.append(traj_key)
                            print(f"  {traj_key}: {images.shape[0]} frames")
                except Exception as e:
                    print(f"  [Warning] Failed to load {traj_key}/{camera_type}: {e}")

            if not all_images:
                print(f"  No valid images found for {camera_type}")
                continue

            # Create video for this camera
            video_filename = f"{camera_type}.{video_format}"
            video_path = output_path / video_filename

            # Get video properties from first image
            sample_image = all_images[0][0]
            height, width = sample_image.shape[:2]

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") if video_format == "mp4" else cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            total_frames = 0

            # Write all images to video
            for traj_idx, images in enumerate(all_images):
                print(f"  Writing trajectory {valid_trajs[traj_idx]} ({len(images)} frames)")

                for frame_idx, image in enumerate(images):
                    # Convert RGB to BGR for OpenCV
                    if image.shape[-1] == 3:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image

                    # Add trajectory and frame info
                    text = f"Traj: {valid_trajs[traj_idx]} Frame: {frame_idx}"
                    cv2.putText(image_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    video_writer.write(image_bgr)
                    total_frames += 1

            video_writer.release()
            print(f"  Created video: {video_path} ({total_frames} total frames)")

            # Create individual trajectory videos
            traj_dir = output_path / f"{camera_type}_trajectories"
            traj_dir.mkdir(exist_ok=True)

            for traj_idx, images in enumerate(all_images):
                traj_video_path = traj_dir / f"{valid_trajs[traj_idx]}.{video_format}"
                traj_writer = cv2.VideoWriter(str(traj_video_path), fourcc, fps, (width, height))

                for frame_idx, image in enumerate(images):
                    if image.shape[-1] == 3:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image

                    # Add frame info
                    text = f"Frame: {frame_idx}"
                    cv2.putText(image_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    traj_writer.write(image_bgr)

                traj_writer.release()
                print(f"    Created trajectory video: {traj_video_path}")


def create_combined_video(h5_path: str, output_dir: str, fps: int = 10, video_format: str = "mp4", max_trajs: int = 5):
    """
    Create a combined video showing all camera views side by side.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        if not traj_keys:
            print("No trajectories found")
            return

        # Randomly select max_trajs trajectories
        if len(traj_keys) > max_trajs:
            selected_trajs = random.sample(traj_keys, max_trajs)
            print(f"Randomly selected {max_trajs} trajectories for combined video: {selected_trajs}")
        else:
            selected_trajs = traj_keys
            print(f"Using all {len(selected_trajs)} available trajectories for combined video")

        # Get camera types
        sample_traj = selected_trajs[0]
        sensor_data_path = f"{sample_traj}/obs/sensor_data"
        camera_types = [k for k in f[sensor_data_path].keys() if "camera" in k]

        if len(camera_types) < 2:
            print("Need at least 2 cameras for combined video")
            return

        print(f"Creating combined video with cameras: {camera_types}")

        # Process each selected trajectory
        for traj_key in selected_trajs:
            try:
                # Load all camera images for this trajectory
                camera_images = {}
                min_frames = float("inf")

                for camera_type in camera_types:
                    image_path = f"{traj_key}/obs/sensor_data/{camera_type}/rgb"
                    if image_path in f:
                        images = f[image_path][:]
                        camera_images[camera_type] = images
                        min_frames = min(min_frames, len(images))

                if not camera_images or min_frames == 0:
                    print(f"No valid images for {traj_key}")
                    continue

                # Get image dimensions
                sample_image = list(camera_images.values())[0][0]
                img_height, img_width = sample_image.shape[:2]

                # Calculate combined video dimensions
                num_cameras = len(camera_images)
                cols = 2 if num_cameras <= 4 else 3
                rows = (num_cameras + cols - 1) // cols

                combined_width = cols * img_width
                combined_height = rows * img_height

                # Create video writer
                video_path = output_path / f"{traj_key}_combined.{video_format}"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v") if video_format == "mp4" else cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (combined_width, combined_height))

                print(f"Creating combined video for {traj_key} ({min_frames} frames)")

                for frame_idx in range(min_frames):
                    # Create combined frame
                    combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

                    for cam_idx, (camera_type, images) in enumerate(camera_images.items()):
                        row = cam_idx // cols
                        col = cam_idx % cols

                        y_start = row * img_height
                        y_end = y_start + img_height
                        x_start = col * img_width
                        x_end = x_start + img_width

                        image = images[frame_idx]
                        if image.shape[-1] == 3:
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        else:
                            image_bgr = image

                        # Add camera label
                        cv2.putText(image_bgr, camera_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        combined_frame[y_start:y_end, x_start:x_end] = image_bgr

                    # Add frame info to combined frame
                    cv2.putText(
                        combined_frame,
                        f"Frame: {frame_idx}",
                        (10, combined_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    video_writer.write(combined_frame)

                video_writer.release()
                print(f"Created combined video: {video_path}")

            except Exception as e:
                print(f"Error processing {traj_key}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert H5 images to videos for visualization")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the videos")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output videos (default: 10)")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "avi"], help="Video format (default: mp4)")
    parser.add_argument("--combined", action="store_true", help="Also create combined videos showing all cameras")
    parser.add_argument(
        "--max_trajs", type=int, default=5, help="Maximum number of trajectories to process (default: 5)"
    )

    args = parser.parse_args()

    print(f"Converting H5 file: {args.h5_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Video settings: {args.fps} FPS, {args.format} format")
    print(f"Max trajectories: {args.max_trajs}")

    # Create individual camera videos
    create_video_from_h5(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        fps=args.fps,
        video_format=args.format,
        max_trajs=args.max_trajs,
    )

    # Create combined videos if requested
    if args.combined:
        print("\nCreating combined videos...")
        create_combined_video(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            fps=args.fps,
            video_format=args.format,
            max_trajs=args.max_trajs,
        )

    print("\nVideo conversion completed!")
