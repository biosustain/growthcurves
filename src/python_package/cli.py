import argparse

from .mockup import hello_world


def main():
    parser = argparse.ArgumentParser(description="Python Package CLI")
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the greeting hello world",
    )
    args = parser.parse_args()
    msg = hello_world(args.repeat)
    print(msg)
