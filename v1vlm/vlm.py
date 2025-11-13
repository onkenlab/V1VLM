import re
import textwrap
from pathlib import Path
from typing import Any

import torch
from markdown_pdf import MarkdownPdf, Section
from transformers import pipeline


class VisionLanguageModel:
    generator: Any
    chat: list[dict[str, Any]]
    context: str

    def __init__(self, context_file) -> None:
        vlm_name = "google/gemma-3-4b-it"
        self.generator = pipeline(
            "image-text-to-text",
            model=vlm_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.chat = []
        with open(context_file, "r") as file:
            self.context = file.read()

    def initialize_chat(self, input_image: Any, response_image: Any) -> None:
        self.chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI scientist specialized in NeuroAI models. \
                              Your task is to come up with new neural coding hypotheses \
                              about static naturalistic image features that do not include \
                              any motion or other temporal characteristics, to test these \
                              hypotheses in silico, and to produce a report of your \
                              hypotheses and findings. \
                              Your input will always include a pair of images, \
                              where the first image is the static input to the in silico model \
                              of a mouse primary visual cortex population \
                              and the second image represents the neuronal responses \
                              produced by the in silico model, where each horizontal \
                              line of the second image represents a recorded activity trace. \
                              There is no representation of time in the static input image. \
                              The input image will presented to the in silico model \
                              for a fixed duration, and the dynamic neuronal responses \
                              for that duration will be represented in the second image. \
                              Your responses will be used to generate new static image input \
                              to be processed by the in silico model. \
                              For this purpose, each of your responses must include a \
                              grayscale image generation prompt enclosed in \
                              <GENERATION_PROMPT> </GENERATION_PROMPT> tags \
                              describing the content of the next static image to test. Please start \
                              your response with this prompt. \
                              The rest of your response should include a brief rationale for \
                              the selected prompt and a couple of sentences describing the \
                              findings that you obtained so far. \
                              For context, consider the following:\n"
                        + self.context,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_image},
                    {"type": "image", "image": response_image},
                    {
                        "type": "text",
                        "text": "Now produce an initial hypothesis about how visual \
                              information is encoded in the primary visual cortex of mice \
                              as well as an image generation prompt for the first input \
                              image to test and refine your hypothesis.",
                    },
                ],
            },
        ]
        output = self.generator(text=self.chat, max_new_tokens=1024)
        self.append_assistant_message(output[0]["generated_text"][-1]["content"])

    def append_assistant_message(self, message: str) -> None:
        self.chat.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": message}],
            }
        )

    def get_last_response(self) -> str:
        return self.chat[-1]["content"][0]["text"]

    def get_image_prompt(self) -> str:
        response = self.get_last_response()
        """Extract the image generation prompt from the model's output."""
        match = re.search(r"<GENERATION_PROMPT>(.*?)</GENERATION_PROMPT>", response)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("No generation prompt found in the output.")

    def process_images(self, input_image: Any, response_image: Any) -> None:
        next_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": input_image},
                {"type": "image", "image": response_image},
                {
                    "type": "text",
                    "text": "Here is the input image and the neuronal response. \
                          Now produce a detailed analysis of the results, \
                          as well as a grayscale image generation prompt for the next \
                          input image to test your hypothesis.",
                },
            ],
        }
        self.chat.append(next_message)
        output = self.generator(text=self.chat, max_new_tokens=1024)
        self.append_assistant_message(output[0]["generated_text"][-1]["content"])

    def produce_final_report(self, save_dir: Path) -> None:
        next_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Now write a full report, including the idea explored, \
                          experiments, results, and conclusions. Interpret these \
                          in light of the context that I mentioned earlier and include \
                          relevant citations from that context. Do not include any \
                          citations not in that context. Also, \
                          suggest future directions for research. Do not include \
                          any image generation prompts in this report.",
                }
            ],
        }
        self.chat.append(next_message)
        output = self.generator(text=self.chat, max_new_tokens=2048)
        self.append_assistant_message(output[0]["generated_text"][-1]["content"])

        # Save chat
        with open(f"{save_dir}/chat.txt", "w") as f:
            f.write(str(self.chat))

        report = self.get_last_response()
        print(report)

        with open(f"{save_dir}/report.md", "w") as f:
            f.write(report)

        pdf = MarkdownPdf()
        pdf.meta["title"] = "Study on Neural Coding Hypotheses in Mouse V1"
        pdf.add_section(Section(report, toc=False))
        pdf.save(f"{save_dir}/report.pdf")
