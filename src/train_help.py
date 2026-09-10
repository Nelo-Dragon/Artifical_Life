import argparse
import random
from pathlib import Path


def generate_addition_data(width: int, operand_bits: int,
                           repeats: int, seed: int) -> list[str]:
    """Generate operand pairs and sums, ordered from simple to difficult."""
    if operand_bits * 2 > width:
        raise ValueError("two operands must fit in the input width")
    limit = 1 << operand_bits
    pairs = [
        (left, right)
        for total in range(2 * (limit - 1) + 1)
        for left in range(limit)
        for right in range(limit)
        if left + right == total
    ]
    curriculum = [
        f"{(left | (right << operand_bits)):0{width}b} : "
        f"{left + right:0{width}b}"
        for left, right in pairs
    ]
    random_generator = random.Random(seed)
    random_pairs = [
        (random_generator.randrange(limit), random_generator.randrange(limit))
        for _ in range(len(curriculum))
    ]
    random_examples = [
        f"{(left | (right << operand_bits)):0{width}b} : "
        f"{left + right:0{width}b}"
        for left, right in random_pairs
    ]
    examples: list[str] = []
    for _ in range(repeats):
        for ordered, random_example in zip(curriculum, random_examples):
            examples.extend((ordered, random_example))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate operand/sum training pairs for the chunk."
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--operand-bits", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0xA17F2026)
    parser.add_argument("--output", type=Path, default=Path("wanted_outputs.txt"))
    args = parser.parse_args()
    if not 1 <= args.width <= 63:
        parser.error("--width must be between 1 and 63")
    if not 1 <= args.operand_bits <= 31:
        parser.error("--operand-bits must be between 1 and 31")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    try:
        training_data = generate_addition_data(
            args.width, args.operand_bits, args.repeats, args.seed
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text("\n".join(training_data) + "\n")
    print(f"Wrote {len(training_data)} arithmetic training pairs to {args.output}")


if __name__ == "__main__":
    main()
