"""
Deepfake Guardian AI - Detection Core: Grad-CAM Explainability
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) as specified in the blueprint:
[Grad-CAM (Explainability) -> Heatmap of Influential Regions]
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class GradCAMExplainer:
    """
    Computes visual explanations for the CNN classifier decisions using Grad-CAM.
    Identifies influential facial regions (blending seams, GAN artifacts, ocular asymmetry).
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.get_target_conv_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self, 
        tensor_input: torch.Tensor, 
        target_class: Optional[int] = None, 
        alpha: float = 0.55
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM class activation heatmap and overlay.
        target_class: 1 for FAKE, 0 for REAL. If None, uses predicted class.
        """
        self.model.eval()
        tensor_input = tensor_input.clone().requires_grad_(True)
        
        # Forward pass
        logits = self.model(tensor_input)
        
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Backward pass for the target class logit
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling on gradients to obtain alpha weights
        # gradients shape: (1, Channels, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # Apply ReLU to retain only positive influence towards class decision
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return {
            "cam_raw": cam,
            "target_class": target_class,
            "class_name": "FAKE" if target_class == 1 else "REAL"
        }

    def overlay_heatmap(
        self, 
        cam_raw: np.ndarray, 
        original_rgb: np.ndarray, 
        colormap: int = cv2.COLORMAP_JET, 
        alpha: float = 0.50
    ) -> Dict[str, np.ndarray]:
        """
        Resizes and blends Grad-CAM heatmap with original RGB face crop.
        Returns pure heatmap and blended overlay.
        """
        h, w = original_rgb.shape[:2]
        
        # Resize raw activation map to image dimensions
        resized_cam = cv2.resize(cam_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        resized_cam = np.clip(resized_cam, 0.0, 1.0)

        # Convert to 8-bit image
        cam_uint8 = np.uint8(255 * resized_cam)

        # Apply color map (JET provides thermal red-to-yellow gradient matching the diagram)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original image
        blended = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

        # Extract top activation coordinates / regions
        high_activation_mask = resized_cam > 0.65
        hotspot_pct = float(np.mean(high_activation_mask) * 100.0)

        return {
            "heatmap_rgb": heatmap_rgb,
            "blended_rgb": blended,
            "activation_map": resized_cam,
            "hotspot_percentage": hotspot_pct
        }
