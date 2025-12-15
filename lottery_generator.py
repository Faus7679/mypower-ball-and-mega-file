#!/usr/bin/env python3
"""
Lottery Number Generator
Generates random numbers for Powerball and Mega Millions games.
"""

import random


def generate_powerball():
    """
    Generate Powerball numbers.
    Rules: 5 unique numbers from 1-69, plus 1 Powerball number from 1-26
    
    Returns:
        tuple: (list of 5 main numbers, powerball number)
    """
    main_numbers = sorted(random.sample(range(1, 70), 5))
    powerball = random.randint(1, 26)
    return main_numbers, powerball


def generate_mega_millions():
    """
    Generate Mega Millions numbers.
    Rules: 5 unique numbers from 1-70, plus 1 Mega Ball number from 1-25
    
    Returns:
        tuple: (list of 5 main numbers, mega ball number)
    """
    main_numbers = sorted(random.sample(range(1, 71), 5))
    mega_ball = random.randint(1, 25)
    return main_numbers, mega_ball


def display_powerball(main_numbers, powerball):
    """Display Powerball numbers in a formatted way."""
    print("\n" + "="*50)
    print("           POWERBALL NUMBERS")
    print("="*50)
    print(f"Main Numbers: {' - '.join(map(str, main_numbers))}")
    print(f"Powerball:    {powerball}")
    print("="*50)


def display_mega_millions(main_numbers, mega_ball):
    """Display Mega Millions numbers in a formatted way."""
    print("\n" + "="*50)
    print("         MEGA MILLIONS NUMBERS")
    print("="*50)
    print(f"Main Numbers: {' - '.join(map(str, main_numbers))}")
    print(f"Mega Ball:    {mega_ball}")
    print("="*50)


def main():
    """Main function to run the lottery generator."""
    print("\n╔════════════════════════════════════════════════╗")
    print("║     LOTTERY NUMBER GENERATOR                   ║")
    print("╚════════════════════════════════════════════════╝")
    
    while True:
        print("\nSelect a game:")
        print("1. Powerball")
        print("2. Mega Millions")
        print("3. Both Games")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            main_numbers, powerball = generate_powerball()
            display_powerball(main_numbers, powerball)
        elif choice == '2':
            main_numbers, mega_ball = generate_mega_millions()
            display_mega_millions(main_numbers, mega_ball)
        elif choice == '3':
            main_numbers, powerball = generate_powerball()
            display_powerball(main_numbers, powerball)
            main_numbers, mega_ball = generate_mega_millions()
            display_mega_millions(main_numbers, mega_ball)
        elif choice == '4':
            print("\nThank you for using Lottery Number Generator!")
            print("Good luck! 🍀\n")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
