import argparse
from pathlib import Path


def generate_identity_data(width: int, repeats: int) -> list[str]:
    """Generate feasible identity targets in increasing signal complexity.

    The chunk is a propagating grid, not an arithmetic unit. Ordering patterns
    by popcount gives the trainer a curriculum: zero, one-bit signals, then
    increasingly complex combinations.
    """
    values = sorted(range(1 << width), key=lambda value: (value.bit_count(), value))
    patterns = [
        f"{value:0{width}b} : {value:0{width}b}"
        for value in values
    ]
    return patterns * repeats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate learnable identity training pairs for the chunk."
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("wanted_outputs.txt"))
    args = parser.parse_args()
    if not 1 <= args.width <= 63:
        parser.error("--width must be between 1 and 63")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    training_data = generate_identity_data(args.width, args.repeats)
    args.output.write_text("\n".join(training_data) + "\n")
    print(f"Wrote {len(training_data)} identity training pairs to {args.output}")


if __name__ == "__main__":
    main()
