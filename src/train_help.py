from pathlib import Path


XS = 16
YS = 16
REPEATS = 4
OUTPUT_FILE = Path("wanted_outputs.txt")


def generate_binary_addition_pair(val1: int, val2: int) -> str:
    if XS % 2 != 0:
        raise ValueError("XS must be even so it can hold two equal-size operands")

    operand_bits = XS // 2
    operand_mask = (1 << operand_bits) - 1
    val1_2 = val1 & operand_mask
    val2_2 = val2 & operand_mask

    # Format the two operands so their combined input is XS bits.
    bin1_2 = f"{val1_2:0{operand_bits}b}"
    bin2_2 = f"{val2_2:0{operand_bits}b}"

    total_sum = val1_2 + val2_2
    sum_bits = f"{total_sum:0{XS}b}"

    return f"{bin1_2}{bin2_2} : {sum_bits}"


def generate_training_data() -> list[str]:
    operand_values = range(1 << (XS // 2))
    combinations = [
        generate_binary_addition_pair(val1, val2)
        for val1 in operand_values
        for val2 in operand_values
    ]
    return combinations * REPEATS


training_data = generate_training_data()
OUTPUT_FILE.write_text("\n".join(training_data) + "\n")
print(f"Wrote {len(training_data)} training pairs to {OUTPUT_FILE}")