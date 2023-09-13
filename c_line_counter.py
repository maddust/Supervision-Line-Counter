from typing import Dict
import cv2
import numpy as np
from supervision.detection.core import Detections
from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect, Vector, Position
from typing import Optional


class custom_LineZone:
    def __init__(self,config,name, start: Point, end: Point, grace_period: int = 60):
        self.config = config
        self.name = name
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.already_counted: Dict[str, bool] = {}  # Flag for each tracker_id
        self.frame_last_seen: Dict[str, int] = {}  # Last frame each tracker_id was encountered
        self.current_frame: int = 0  # Current frame counter
        self.in_count: int = 0
        self.out_count: int = 0
        self.grace_period = grace_period  # Number of frames to wait before resetting
    def is_within_line_segment(self, point: Point, margin: float = 20) -> bool:
        """
        Check if a point is within the line segment.

        Attributes:
            point (Point): The point to be checked.
            margin (float): Additional margin added to the line segment boundaries.

        Returns:
            bool: True if the point is within the line segment, False otherwise.
        """
        x_within = min(self.vector.start.x, self.vector.end.x) - margin <= point.x <= max(self.vector.start.x, self.vector.end.x) + margin
        y_within = min(self.vector.start.y, self.vector.end.y) - margin <= point.y <= max(self.vector.start.y, self.vector.end.y) + margin

        return x_within and y_within


    def trigger(self, detections: Detections):
        self.current_frame += 1  # Increment frame counter

        for xyxy, _, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if the bottom center anchor of bbox is on the same side of vector
            x1, y1, x2, y2 = xyxy
            if self.config["ANCHOR_POINT"] == "center":
                anchor = Point(x=(x1+x2)/2, y=(y1+y2)/2)  # Center point of bounding box
            else:  # Default to bottom_center
                anchor = Point(x=(x1+x2)/2, y=y2)  # Bottom center point of bounding box
            # Check if anchor is within the line segment
            if not self.is_within_line_segment(anchor):
                continue

            tracker_state = self.vector.is_in(point=anchor)

            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                self.already_counted[tracker_id] = False  # When the object appears, it has not been counted yet
                self.frame_last_seen[tracker_id] = self.current_frame  # Update last seen frame
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            # check if this crossing has already been counted
            if self.already_counted.get(tracker_id, False):  # If it has been counted, we skip this
                # Check if grace period has passed since last encounter
                if self.current_frame - self.frame_last_seen.get(tracker_id, 0) > self.grace_period:
                    self.already_counted[tracker_id] = False  # Reset after grace period
                else:
                    continue

            self.tracker_state[tracker_id] = tracker_state
            self.already_counted[tracker_id] = True  # After counting, we mark this crossing as already counted
            self.frame_last_seen[tracker_id] = self.current_frame  # Update last seen frame

            if tracker_state:
                self.in_count += 1
            else:
                self.out_count += 1



class custom_LineZoneAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: Optional[str] = None,
        custom_out_text: Optional[str] = None,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        Attributes:
            thickness (float): The thickness of the line that will be drawn.
            color (Color): The color of the line that will be drawn.
            text_thickness (float): The thickness of the text that will be drawn.
            text_color (Color): The color of the text that will be drawn.
            text_scale (float): The scale of the text that will be drawn.
            text_offset (float): The offset of the text that will be drawn.
            text_padding (int): The padding of the text that will be drawn.

        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.custom_in_text: str = custom_in_text
        self.custom_out_text: str = custom_out_text

    def annotate(self, frame: np.ndarray, line_counter: custom_LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineCounter): The line counter that will be used to draw the line.

        Returns:
            np.ndarray: The image with the line drawn on it.

        """
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=self.color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        in_text = f"{self.custom_in_text}: {line_counter.in_count}" if self.custom_in_text is not None else f"in: {line_counter.in_count}"
        out_text = f"{self.custom_out_text}: {line_counter.out_count}" if self.custom_out_text is not None else f"out: {line_counter.out_count}"


        (in_text_width, in_text_height), _ = cv2.getTextSize(
            in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        in_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - in_text_width)
            / 2
        )
        in_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + in_text_height)
            / 2
            - self.text_offset * in_text_height
        )

        out_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width)
            / 2
        )
        out_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height)
            / 2
            + self.text_offset * out_text_height
        )

        in_text_background_rect = Rect(
            x=in_text_x,
            y=in_text_y - in_text_height,
            width=in_text_width,
            height=in_text_height,
        ).pad(padding=self.text_padding)
        out_text_background_rect = Rect(
            x=out_text_x,
            y=out_text_y - out_text_height,
            width=out_text_width,
            height=out_text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            frame,
            in_text_background_rect.top_left.as_xy_int_tuple(),
            in_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )
        cv2.rectangle(
            frame,
            out_text_background_rect.top_left.as_xy_int_tuple(),
            out_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            in_text,
            (in_text_x, in_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        return frame
