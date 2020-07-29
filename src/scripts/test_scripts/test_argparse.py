import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=str, required=True)
parser.add_argument('--b', type=str, required=True)
parser.add_argument('-c', type=bool, default=False, const=True, nargs='?')
parser.add_argument('--d', type=int, default=42, required=False)
arguments = parser.parse_args()

print(arguments.c)