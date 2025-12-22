import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import Any

from v1vlm.v1vlm import V1VLM


def main(args: Any) -> None:
    model = V1VLM(args)
    model.run_study(args.num_steps)


if __name__ == "__main__":
    now = datetime.now()
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default="./data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./runs/viv1t",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        default="./context.txt",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default="./study_" + now.strftime("%Y%m%d%H%M%S"),
    )
    parser.add_argument(
        "--vlm-size",
        type=str,
        default="4b",
        choices=["4b", "12b"],
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default="",
    )
    parser.add_argument("--mouse_id", type=str, default="A")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument(
        "--input-generator-device",
        type=str,
        default="cpu",
        help="Device to use for input generation.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the digital twin"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "bf16"],
        default="bf16",
        help="ViV1T precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=5, help="Number of interaction steps"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
