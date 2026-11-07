"""
Deepfake Guardian AI - Preprocessing Layer (OpenCV)
Implements Face Detection, Cropping, Resizing, and PyTorch Tensor Normalization.
Matches the architectural blueprint:
[Input Layer] -> [Preprocessing Layer (OpenCV: Face Detection, Cropping, Resizing, Normalization)]
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from typing import Tuple, Dict, Any, Optional

import config

class PreprocessingLayer:
    """OpenCV-based preprocessing pipeline for facial deepfake detection."""

    def __init__(self, target_size: Tuple[int, int] = config.INPUT_IMAGE_SIZE):
        self.target_size = target_size
        
        # Load Haar Cascade Face Detector from OpenCV's built-in data
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # PyTorch normalization transform
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZATION_MEAN, 
                std=config.NORMALIZATION_STD
            )
        ])

    def detect_face_bbox(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary frontal face bounding box (x, y, w, h).
        Returns the largest detected face.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            # Fallback with relaxed parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(40, 40)
            )

        if len(faces) == 0:
            return None

        # Return the largest face by area
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        return tuple(largest_face)

    def process(self, input_image: Any) -> Dict[str, Any]:
        """
        Executes the full preprocessing pipeline:
        1. Face Detection
        2. Cropping (with contextual margin)
        3. Resizing (to target_size)
        4. Normalization (to PyTorch tensor)
        """
        # Convert input to BGR numpy array
        if isinstance(input_image, Image.Image):
            input_image = ImageOps.exif_transpose(input_image)
            rgb_arr = np.array(input_image.convert("RGB"))
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 2:
                bgr_arr = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            elif input_image.shape[2] == 4:
                bgr_arr = cv2.cvtColor(input_image, cv2.COLOR_BGRA2BGR)
            else:
                bgr_arr = input_image.copy()
        else:
            raise ValueError("Unsupported image type. Provide PIL.Image or np.ndarray.")

        h, w = bgr_arr.shape[:2]
        bbox = self.detect_face_bbox(bgr_arr)

        annotated_bgr = bgr_arr.copy()
        face_detected = bbox is not None

        if face_detected:
            fx, fy, fw, fh = bbox
            # Add 15% contextual padding around face to capture blending boundaries
            pad_x = int(fw * 0.15)
            pad_y = int(fh * 0.15)

            x1 = max(0, fx - pad_x)
            y1 = max(0, fy - pad_y)
            x2 = min(w, fx + fw + pad_x)
            y2 = min(h, fy + fh + pad_y)

            face_crop_bgr = bgr_arr[y1:y2, x1:x2]

            # Draw green bounding box matching architecture visual
            cv2.rectangle(annotated_bgr, (fx, fy), (fx + fw, fy + fh), (0, 255, 70), 3)
            # Add tech corner marks
            c_len = int(min(fw, fh) * 0.15)
            cv2.line(annotated_bgr, (fx, fy), (fx + c_len, fy), (0, 255, 70), 5)
            cv2.line(annotated_bgr, (fx, fy), (fx, fy + c_len), (0, 255, 70), 5)
            cv2.line(annotated_bgr, (fx + fw, fy), (fx + fw - c_len, fy), (0, 255, 70), 5)
            cv2.line(annotated_bgr, (fx + fw, fy), (fx + fw, fy + c_len), (0, 255, 70), 5)
            cv2.putText(
                annotated_bgr, "FACE LOCK: VERIFIED", 
                (fx, max(20, fy - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 70), 2
            )
        else:
            # Fallback: crop center square if no face is detected
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            face_crop_bgr = bgr_arr[start_y:start_y+min_dim, start_x:start_x+min_dim]
            bbox = (start_x, start_y, min_dim, min_dim)
            cv2.rectangle(annotated_bgr, (start_x, start_y), (start_x + min_dim, start_y + min_dim), (255, 165, 0), 2)
            cv2.putText(
                annotated_bgr, "SCAN: FULL FRAME FOCUS", 
                (start_x, max(20, start_y - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2
            )

        # 3. Resizing
        resized_bgr = cv2.resize(face_crop_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        # 4. Normalization (PyTorch Tensor)
        pil_resized = Image.fromarray(resized_rgb)
        tensor_normalized = self.tensor_transform(pil_resized).unsqueeze(0)  # Shape: (1, 3, 224, 224)

        return {
            "face_detected": face_detected,
            "bbox": bbox,
            "original_annotated_rgb": original_rgb,
            "crop_rgb": crop_rgb,
            "resized_rgb": resized_rgb,
            "normalized_tensor": tensor_normalized,
            "dimensions": {
                "original": (w, h),
                "crop": (crop_rgb.shape[1], crop_rgb.shape[0]),
                "target": self.target_size
            }
        }
