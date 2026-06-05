import torch
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        self.model.layer4.register_forward_hook(
            self._save_activations)
        self.model.layer4.register_full_backward_hook(
            self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, tensor, class_idx=None):
        self.model.eval()

        output = self.model(tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients   = self.gradients.squeeze().cpu().detach().numpy()
        activations = self.activations.squeeze().cpu().detach().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam     = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam

    def overlay(self, cam, original_image, alpha=0.5):
        img        = np.array(original_image.resize((224, 224)))
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap    = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap    = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlaid   = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
        return Image.fromarray(overlaid)