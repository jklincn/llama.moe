import sys

from .wrapper import LlamaServerWrapper


def main():
    wrapper = LlamaServerWrapper()
    return wrapper.run()


if __name__ == "__main__":
    sys.exit(main())
