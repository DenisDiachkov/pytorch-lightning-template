from options.options import parse_args
from test import test
from train import train
import warnings


def main():
    args = parse_args()
    if args.Wall:
        warnings.simplefilter("error")
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)


if __name__ == "__main__":
    main()
