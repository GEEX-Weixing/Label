from src.argument import parse_args
from src.utils import set_random_seeds
from model import CL_Trainer_C
import torch


def main():
    args = parse_args()
    set_random_seeds(42)
    torch.set_num_threads(4)

    embedder = CL_Trainer_C(args)
    embedder.train()


if __name__ == "__main__":
    main()