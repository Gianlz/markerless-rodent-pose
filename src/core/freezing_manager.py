"""Freezing behavior analysis manager"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import logging

from .inference_manager import InferenceManager

logger = logging.getLogger(__name__)


class FreezingManager:
    """
    Manages the analysis of freezing behavior and side preference tests.
    """

    def __init__(self):
        self.inference_manager = InferenceManager()

    def run_freezing_analysis(
        self,
        video_path: str,
        config_path: str,
        line_points: List[Tuple[int, int]],
        force_analysis: bool = False,
        create_video: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Runs the full freezing analysis pipeline.

        Args:
            video_path: Path to the video file.
            config_path: Path to the DeepLabCut config.yaml.
            line_points: List of two (x, y) tuples defining the center line.
            force_analysis: If True, re-runs DeepLabCut analysis.
            create_video: If True, generates a labeled video.
            progress_callback: Optional callback for status updates.

        Returns:
            Path to the generated Excel report.
        """
        if progress_callback:
            progress_callback("Checking analysis status...")

        video_path_obj = Path(video_path)

        # Check if analysis is needed
        analysis_exists = self.inference_manager.check_analysis_exists(
            str(video_path_obj), config_path
        )

        if force_analysis or not analysis_exists:
            if progress_callback:
                progress_callback("Running DeepLabCut analysis (this may take time)...")
            self.inference_manager.analyze_videos(
                config_path, [str(video_path_obj)], save_as_csv=True
            )

        if progress_callback:
            progress_callback("Loading data...")

        # Find analysis file
        h5_files = list(video_path_obj.parent.glob(f"{video_path_obj.stem}DLC*.h5"))
        if not h5_files:
            raise FileNotFoundError(
                "No analysis file found even after running analysis."
            )

        # Use the most recent one if multiple
        h5_file = sorted(h5_files)[-1]

        df = pd.read_hdf(h5_file)

        # Get scorer and bodyparts
        scorer = df.columns.levels[0][0]
        bodyparts = df.columns.levels[1]

        # Check for required bodyparts
        required_paws = ["FR_paw", "FL_paw", "BR_paw", "BL_paw"]
        for paw in required_paws:
            if paw not in bodyparts:
                raise ValueError(
                    f"Missing bodypart: {paw}. Model must have: {required_paws}"
                )

        if progress_callback:
            progress_callback("Analyzing frames...")

        # Define line
        p1 = np.array(line_points[0])
        p2 = np.array(line_points[1])

        # Vector representing the line
        line_vec = p2 - p1

        # Normal vector to the line (for determining side)
        # Rotate 90 degrees
        normal_vec = np.array([-line_vec[1], line_vec[0]])

        def get_side(point):
            # Dot product with normal vector determines side
            # point is (x, y)
            vec_to_point = point - p1
            return np.sign(np.dot(vec_to_point, normal_vec))

        # Open Video Capture
        cap = cv2.VideoCapture(str(video_path_obj))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not fps or np.isnan(fps):
            fps = 30.0  # Fallback

        # Setup Video Writer if requested
        writer = None
        output_video_path = None
        if create_video:
            output_video_path = (
                video_path_obj.parent / f"{video_path_obj.stem}_labeled.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )

        total_frames = len(df)
        side_a_frames = 0
        side_b_frames = 0

        # Event logging
        events = []
        current_state = None  # 'A', 'B'
        state_start_frame = 0
        first_detection_frame = None

        try:
            # Iterate through frames
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                paws_on_side_a = 0
                paws_on_side_b = 0

                # Draw center line on frame
                if writer:
                    cv2.line(
                        frame,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        (0, 0, 255),  # Red
                        2,
                    )

                for paw in required_paws:
                    x = df.iloc[i][(scorer, paw, "x")]
                    y = df.iloc[i][(scorer, paw, "y")]
                    likelihood = df.iloc[i][(scorer, paw, "likelihood")]

                    if likelihood < 0.1:
                        continue

                    side = get_side(np.array([x, y]))
                    if side > 0:
                        paws_on_side_a += 1
                    else:
                        paws_on_side_b += 1

                    # Draw paw points
                    if writer:
                        color = (
                            (0, 255, 0) if side > 0 else (255, 0, 0)
                        )  # Green for A, Blue for B
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

                # Determine detected state for this frame (Instantaneous)
                detected_state = None
                if paws_on_side_a == 4:
                    detected_state = "A"
                elif paws_on_side_b == 4:
                    detected_state = "B"

                # State Machine Logic
                if current_state is None:
                    # Initialize state if we have a definitive detection
                    if detected_state is not None:
                        current_state = detected_state
                        state_start_frame = i
                        if first_detection_frame is None:
                            first_detection_frame = i
                else:
                    # We have a current state, check for transition
                    # Only transition if we strictly detect the OTHER side (4 paws)
                    if detected_state is not None and detected_state != current_state:
                        # State changed!
                        # Log previous event
                        duration_frames = i - state_start_frame
                        events.append(
                            {
                                "Start Time (s)": state_start_frame / fps,
                                "End Time (s)": i / fps,
                                "Duration (s)": duration_frames / fps,
                                "Side": current_state,
                            }
                        )

                        # Update state
                        current_state = detected_state
                        state_start_frame = i

                # Accumulate frames based on PERSISTENT state
                if current_state == "A":
                    side_a_frames += 1
                elif current_state == "B":
                    side_b_frames += 1

                # Write frame with overlay
                if writer:
                    # Draw Side Label
                    label_text = (
                        f"Side: {current_state if current_state else 'Unknown'}"
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),  # Yellow
                        3,
                    )

                    # Draw Stats
                    stats_text = f"A: {side_a_frames} | B: {side_b_frames}"
                    cv2.putText(
                        frame,
                        stats_text,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    writer.write(frame)

                # Update progress every 100 frames
                if i % 100 == 0 and progress_callback:
                    progress_callback(f"Processing frame {i}/{total_frames}...")

        finally:
            cap.release()
            if writer:
                writer.release()

        # Close last event
        if current_state is not None:
            duration_frames = total_frames - state_start_frame
            events.append(
                {
                    "Start Time (s)": state_start_frame / fps,
                    "End Time (s)": total_frames / fps,
                    "Duration (s)": duration_frames / fps,
                    "Side": current_state,
                }
            )

        side_a_time = side_a_frames / fps
        side_b_time = side_b_frames / fps
        # Prepare output
        output_path = (
            video_path_obj.parent / f"{video_path_obj.stem}_freezing_test.xlsx"
        )

        with pd.ExcelWriter(output_path) as writer:
            # Calculate effective total time (from first detection to end)
            if first_detection_frame is not None:
                effective_total_time = (total_frames - first_detection_frame) / fps
            else:
                effective_total_time = 0.0

            # Summary Data
            summary_data = {
                "Video": [video_path_obj.name],
                "Side A Total Time (s)": [side_a_time],
                "Side B Total Time (s)": [side_b_time],
                "Total Video Time (s)": [effective_total_time],
                "FPS": [fps],
            }
            summary_df = pd.DataFrame(summary_data)

            # Events Data
            if events:
                events_df = pd.DataFrame(events)
            else:
                events_df = pd.DataFrame({"Message": ["No events detected"]})

            # Write to single sheet "Analysis"
            # Summary at the top
            summary_df.to_excel(writer, sheet_name="Analysis", index=False, startrow=0)

            # Detailed Events below
            events_df.to_excel(writer, sheet_name="Analysis", index=False, startrow=3)

        return str(output_path)
