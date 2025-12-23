#!/usr/bin/env python3
"""
Multi Match Lottery Game

Provides a Multi Match-style lottery generator with a higher chance
of matching some numbers than single-line games like Powerball,
thanks to a smaller number pool and multiple lines per ticket.

Note: This is for entertainment only and does NOT guarantee wins.
All numbers are still generated randomly.
"""

import random


class MultiMatchGenerator:
    """Generate random tickets for a Multi Match-style lottery game."""

    def __init__(self, pool_size: int = 43, numbers_per_line: int = 6, lines_per_ticket: int = 3) -> None:
        """Initialize the generator configuration.

        Args:
            pool_size: Highest number in the pool (1..pool_size).
            numbers_per_line: How many unique numbers per line.
            lines_per_ticket: How many lines per ticket.
        """
        self.pool_size = pool_size
        self.numbers_per_line = numbers_per_line
        self.lines_per_ticket = lines_per_ticket

    def generate_ticket(self) -> dict:
        """Generate a Multi Match ticket with multiple lines.

        Returns:
            dict: Contains 'lines', a list of lists of ints.
        """
        lines = []
        for _ in range(self.lines_per_ticket):
            line = sorted(random.sample(range(1, self.pool_size + 1), self.numbers_per_line))
            lines.append(line)
        return {"lines": lines}

    def format_ticket(self, ticket: dict) -> str:
        """Format a Multi Match ticket for display."""
        formatted_lines = []
        for idx, line in enumerate(ticket["lines"], start=1):
            numbers_str = " - ".join(str(n) for n in line)
            formatted_lines.append(f"Line {idx}: {numbers_str}")
        return "\n".join(formatted_lines)


def main() -> None:
    """Run a simple CLI for generating a Multi Match ticket."""
    generator = MultiMatchGenerator()
    print("=" * 60)
    print("Multi Match Lottery Ticket".center(60))
    print("=" * 60)
    print()

    ticket = generator.generate_ticket()
    print(generator.format_ticket(ticket))


if __name__ == "__main__":
    main()
