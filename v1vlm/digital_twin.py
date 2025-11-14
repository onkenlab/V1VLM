import logging
from pathlib import Path
from typing import Any

import torch
from torchvision import transforms

from viv1t import data
from viv1t.data.data import MovieDataset
from viv1t.model import Model
from viv1t.utils import utils
from viv1t.utils.load_model import load_model

SKIP = 50  # skip the first 50 frames from each trial

VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2
FPS = 30

PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class DigitalTwin:
    model: torch.nn.Module
    dataset: MovieDataset
    save_dir: Path
    args: Any

    def __init__(self, args: Any) -> None:
        utils.set_random_seed(args.seed)
        args.device = utils.get_device(args.device)
        self.model, ds = load_model(args, evaluate=False, compile=args.compile)
        self.dataset = ds[args.mouse_id].dataset
        self.args = args
        self.model.train(False)

        self.save_dir = args.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        utils.set_logger_handles(
            logger=logger,
            filename=self.save_dir / f"output-{utils.get_timestamp()}.log",
            level=logging.INFO,
        )

    def make_inputs(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert image.shape == (VIDEO_H, VIDEO_W)

        behavior, pupil_center = data.get_mean_behaviors(
            self.args.mouse_id, num_frames=data.MAX_FRAME
        )
        behavior = self.dataset.transform_behavior(behavior)
        pupil_center = self.dataset.transform_pupil_center(pupil_center)

        blank = torch.full((1, BLANK_SIZE, VIDEO_H, VIDEO_W), fill_value=GREY_COLOR)

        image = transforms.functional.convert_image_dtype(image, torch.uint8)
        img = image.repeat(1, PATTERN_SIZE, 1, 1)
        assert img.shape == (1, PATTERN_SIZE, VIDEO_H, VIDEO_W)
        video = torch.cat([blank, img, blank], dim=1)
        return video, behavior, pupil_center

    @torch.inference_mode()
    def inference(
        self,
        video: torch.Tensor,
        behavior: torch.Tensor,
        pupil_center: torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = self.model.device, self.model.dtype

        response, _ = self.model(
            inputs=video.to(device, dtype),
            mouse_id=self.args.mouse_id,
            behaviors=behavior.to(device, dtype),
            pupil_centers=pupil_center.to(device, dtype),
        )
        response = response.to(torch.float32).detach().cpu()
        return response

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.squeeze(0)
        video, behavior, pupil_center = self.make_inputs(image)

        # Add batch dimension
        video = video.unsqueeze(0)
        behavior = behavior.unsqueeze(0)
        pupil_center = pupil_center.unsqueeze(0)

        response = self.inference(
            video=video,
            behavior=behavior,
            pupil_center=pupil_center,
        )
        # We'll keep just the presentation interval
        response = response.squeeze(0)[:, BLANK_SIZE : (BLANK_SIZE + PATTERN_SIZE)]
        return response
