import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Any

from v1vlm.v1vlm import V1VLM


def main(args: Any) -> None:
    model = V1VLM(args)
    model.run_study(args.num_steps)


if __name__ == "__main__":
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
        default="./study_results",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the model on the validation set after loading the checkpoint.",
    )
    parser.add_argument("--mouse_id", type=str, default="A")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "use the best available device if --device is not specified.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "bf16"],
        default="32",
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=5, help="Number of interaction steps"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
