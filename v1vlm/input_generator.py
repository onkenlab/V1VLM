from typing import Any

import torch
from diffusers import FluxPipeline
from torchvision import transforms


class InputGenerator:
    generator: FluxPipeline
    device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.generator = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell"
        ).to(self.device)

    def generate(self, prompt: str) -> torch.Tensor:
        """Generate the next image based on the given prompt using the provided pipeline."""
        image = self.generator(
            prompt,
            height=384,
            width=512,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
        ).images[0]

        # Crop and resize image
        transform = transforms.Compose(
            [
                transforms.Resize([48, 64]),
                transforms.CenterCrop([36, 64]),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        return transform(image)
