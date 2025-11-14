from pathlib import Path
from typing import Any

import torch
from torchvision import transforms

from v1vlm.digital_twin import DigitalTwin
from v1vlm.input_generator import InputGenerator
from v1vlm.vlm import VisionLanguageModel


class V1VLM:
    args: Any
    digital_twin: DigitalTwin
    vlm: VisionLanguageModel
    input_generator: InputGenerator

    def __init__(
        self,
        args: Any,
    ) -> None:
        self.args = args
        self.digital_twin = DigitalTwin(args)
        self.input_generator = InputGenerator(args.input_generator_device)
        self.vlm = VisionLanguageModel(args.context_file)

    def run_study(self, num_steps: int) -> None:
        initial_image_prompt = "A grayscale image of random noise."
        input_image, response_image = self.run_experiment(initial_image_prompt)
        self.vlm.initialize_chat(input_image, response_image, self.args.initial_prompt)
        save_dir = self.args.save_dir
        print(self.vlm.get_last_response())
        for step in range(num_steps):
            input_image.save(f"{save_dir}/input_image_{step}.png")
            response_image.save(f"{save_dir}/response_image_{step}.png")
            image_prompt = self.vlm.get_image_prompt()
            input_image, response_image = self.run_experiment(image_prompt)
            self.vlm.process_images(input_image, response_image)
            print(self.vlm.get_last_response())
        self.vlm.produce_final_report(save_dir)
        print(self.vlm.get_last_response())

    def run_experiment(self, prompt: str) -> None:
        input_image = self.input_generator.generate(prompt)
        response_tensor = self.digital_twin.process_image(input_image)
        # Convert image tensors to images
        input_image = transforms.ToPILImage()(input_image)
        response_image = transforms.ToPILImage()(response_tensor)
        return input_image, response_image
