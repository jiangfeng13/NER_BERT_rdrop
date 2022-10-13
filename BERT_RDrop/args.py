import argparse
import os


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="msra", type=str)
    args = parser.parse_args()


    return args
