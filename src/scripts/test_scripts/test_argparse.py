import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=str, required=True)
parser.add_argument('--b', type=str, required=True)
parser.add_argument('--c', type=int, required=True)
parser.add_argument('--d', type=int, default=42, required=False)
arguments = parser.parse_args()

print(arguments.d)