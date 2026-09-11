import argparse
import random
from pathlib import Path


def generate_memory_keys(width: int, key_count: int, seed: int) -> list[str]:
    """
    Generate random bit patterns to use as memory retention test keys.
    Each pattern is scored on how well it can be reconstructed after
    a silence phase in the memory chunk's write/silence/probe cycle.
    """
    rng = random.Random(seed)
    keys = []
    for _ in range(key_count):
        # Each key is a random bit pattern of `width` bits
        key_value = rng.randrange(1 << width)
        key_binary = f"{key_value:0{width}b}"
        keys.append(key_binary)
    return keys


def generate_identity_tasks(width: int, task_count: int, seed: int) -> list[str]:
    """
    Generate simple identity-map I/O pairs for pipeline cortex/output stages.
    Format: "input_bits : output_bits" (same value).
    Useful as a fallback when full logic tasks aren't defined yet.
    """
    rng = random.Random(seed)
    tasks = []
    for _ in range(task_count):
        value = rng.randrange(1 << width)
        value_binary = f"{value:0{width}b}"
        tasks.append(f"{value_binary} : {value_binary}")
    return tasks


def generate_bitwise_logic_tasks(
    width: int, task_count: int, operation: str, seed: int
) -> list[str]:
    """
    Generate two-input bitwise logic tasks for pipeline cortex.
    Operations: 'and', 'or', 'xor', 'nand', 'nor'.
    Format: "input (split into left/right halves) : output".
    """
    if width % 2 != 0:
        raise ValueError("width must be even for two-operand tasks")
    half_width = width // 2
    limit = 1 << half_width
    rng = random.Random(seed)
    
    ops = {
        'and': lambda a, b: a & b,
        'or': lambda a, b: a | b,
        'xor': lambda a, b: a ^ b,
        'nand': lambda a, b: (~(a & b)) & ((1 << half_width) - 1),
        'nor': lambda a, b: (~(a | b)) & ((1 << half_width) - 1),
    }
    if operation not in ops:
        raise ValueError(f"unknown operation: {operation}")
    op_func = ops[operation]
    
    tasks = []
    for _ in range(task_count):
        left = rng.randrange(limit)
        right = rng.randrange(limit)
        # Pack left and right into a single input word (left in lower bits)
        combined_input = left | (right << half_width)
        result = op_func(left, right)
        input_binary = f"{combined_input:0{width}b}"
        output_binary = f"{result:0{half_width}b}".ljust(width, '0')
        tasks.append(f"{input_binary} : {output_binary}")
    return tasks


def generate_jittered_patterns(
    patterns: list[str], jitter_density: float, jitter_drop: float, seed: int
) -> list[str]:
    """
    Add bit-flip noise and drop events to patterns for robustness testing.
    jitter_density: probability each bit flips independently.
    jitter_drop: probability an entire cycle is dropped (set to 0).
    """
    rng = random.Random(seed)
    jittered = []
    for pattern in patterns:
        # Split "input : output" format
        if ' : ' in pattern:
            input_bits, output_bits = pattern.split(' : ')
        else:
            input_bits = pattern
            output_bits = ""
        
        # Jitter input
        if rng.random() < jitter_drop:
            jittered_input = '0' * len(input_bits)
        else:
            input_bits_list = list(input_bits)
            for i in range(len(input_bits_list)):
                if rng.random() < jitter_density:
                    input_bits_list[i] = '0' if input_bits_list[i] == '1' else '1'
            jittered_input = ''.join(input_bits_list)
        
        if output_bits:
            # Jitter output similarly if present
            if rng.random() < jitter_drop:
                jittered_output = '0' * len(output_bits)
            else:
                output_bits_list = list(output_bits)
                for i in range(len(output_bits_list)):
                    if rng.random() < jitter_density:
                        output_bits_list[i] = '0' if output_bits_list[i] == '1' else '1'
                jittered_output = ''.join(output_bits_list)
            jittered.append(f"{jittered_input} : {jittered_output}")
        else:
            jittered.append(jittered_input)
    return jittered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training patterns for pipeline stages (memory, I/O, cortex)."
    )
    subparsers = parser.add_subparsers(dest='mode', help='Pattern generation mode')
    
    # Memory retention patterns
    mem_parser = subparsers.add_parser('memory', help='Generate memory keys')
    mem_parser.add_argument('--width', type=int, default=4,
                            help='Bit width of each memory key')
    mem_parser.add_argument('--keys', type=int, default=10,
                            help='Number of distinct keys to generate')
    mem_parser.add_argument('--repeats', type=int, default=1,
                            help='Repeat each key N times in output')
    mem_parser.add_argument('--seed', type=int, default=0xDEADBEEF)
    mem_parser.add_argument('--output', type=Path, default=Path('memory_keys.txt'))
    
    # Identity task
    id_parser = subparsers.add_parser('identity', help='Generate identity I/O tasks')
    id_parser.add_argument('--width', type=int, default=4,
                           help='Bit width of I/O')
    id_parser.add_argument('--tasks', type=int, default=10,
                           help='Number of I/O pairs')
    id_parser.add_argument('--repeats', type=int, default=1,
                           help='Repeat each task N times')
    id_parser.add_argument('--seed', type=int, default=0xDEADBEEF)
    id_parser.add_argument('--output', type=Path, default=Path('identity_tasks.txt'))
    
    # Logic tasks
    logic_parser = subparsers.add_parser('logic', help='Generate bitwise logic tasks')
    logic_parser.add_argument('--width', type=int, default=8,
                              help='Bit width (even), split into two operands')
    logic_parser.add_argument('--operation', type=str, default='xor',
                              choices=['and', 'or', 'xor', 'nand', 'nor'],
                              help='Bitwise operation')
    logic_parser.add_argument('--tasks', type=int, default=20,
                              help='Number of I/O pairs')
    logic_parser.add_argument('--repeats', type=int, default=1,
                              help='Repeat each task N times')
    logic_parser.add_argument('--seed', type=int, default=0xDEADBEEF)
    logic_parser.add_argument('--output', type=Path, default=Path('logic_tasks.txt'))
    
    # Jitter option (shared across all modes)
    for p in [mem_parser, id_parser, logic_parser]:
        p.add_argument('--jitter-density', type=float, default=0.0,
                       help='Bit-flip probability per bit')
        p.add_argument('--jitter-drop', type=float, default=0.0,
                       help='Probability entire cycle is dropped (set to 0)')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    if args.mode == 'memory':
        keys = generate_memory_keys(args.width, args.keys, args.seed)
        patterns = []
        for _ in range(args.repeats):
            patterns.extend(keys)
        if args.jitter_density > 0 or args.jitter_drop > 0:
            patterns = generate_jittered_patterns(
                patterns, args.jitter_density, args.jitter_drop, args.seed + 1
            )
        args.output.write_text('\n'.join(patterns) + '\n')
        print(f"Wrote {len(patterns)} memory key patterns to {args.output}")
    
    elif args.mode == 'identity':
        tasks = generate_identity_tasks(args.width, args.tasks, args.seed)
        patterns = []
        for _ in range(args.repeats):
            patterns.extend(tasks)
        if args.jitter_density > 0 or args.jitter_drop > 0:
            patterns = generate_jittered_patterns(
                patterns, args.jitter_density, args.jitter_drop, args.seed + 1
            )
        args.output.write_text('\n'.join(patterns) + '\n')
        print(f"Wrote {len(patterns)} identity I/O tasks to {args.output}")
    
    elif args.mode == 'logic':
        try:
            tasks = generate_bitwise_logic_tasks(
                args.width, args.tasks, args.operation, args.seed
            )
        except ValueError as e:
            parser.error(str(e))
        patterns = []
        for _ in range(args.repeats):
            patterns.extend(tasks)
        if args.jitter_density > 0 or args.jitter_drop > 0:
            patterns = generate_jittered_patterns(
                patterns, args.jitter_density, args.jitter_drop, args.seed + 1
            )
        args.output.write_text('\n'.join(patterns) + '\n')
        print(f"Wrote {len(patterns)} {args.operation} I/O tasks to {args.output}")


if __name__ == "__main__":
    main()
