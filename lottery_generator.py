#!/usr/bin/env python3
"""
Lottery Number Generator
Generates random numbers for Powerball and Mega Millions games
"""

import random


class LotteryGenerator:
    """Generate random lottery numbers for Powerball and Mega Millions games."""
    
    def generate_powerball(self):
        """
        Generate Powerball numbers.
        Returns 5 unique numbers from 1-69 and 1 Powerball number from 1-26.
        
        Returns:
            dict: Contains 'numbers' (list of 5 ints) and 'powerball' (int)
        """
        numbers = sorted(random.sample(range(1, 70), 5))
        powerball = random.randint(1, 26)
        return {
            'numbers': numbers,
            'powerball': powerball
        }
    
    def generate_mega_millions(self):
        """
        Generate Mega Millions numbers.
        Returns 5 unique numbers from 1-70 and 1 Mega Ball number from 1-25.
        
        Returns:
            dict: Contains 'numbers' (list of 5 ints) and 'mega_ball' (int)
        """
        numbers = sorted(random.sample(range(1, 71), 5))
        mega_ball = random.randint(1, 25)
        return {
            'numbers': numbers,
            'mega_ball': mega_ball
        }
    
    def format_powerball(self, result):
        """Format Powerball result for display."""
        numbers_str = ' - '.join(str(n) for n in result['numbers'])
        return f"Numbers: {numbers_str} | Powerball: {result['powerball']}"
    
    def format_mega_millions(self, result):
        """Format Mega Millions result for display."""
        numbers_str = ' - '.join(str(n) for n in result['numbers'])
        return f"Numbers: {numbers_str} | Mega Ball: {result['mega_ball']}"


def main():
    """Main function to run the lottery number generator."""
    generator = LotteryGenerator()
    
    print("=" * 60)
    print("Lottery Number Generator".center(60))
    print("=" * 60)
    print()
    
    while True:
        print("\nSelect a game:")
        print("1. Powerball")
        print("2. Mega Millions")
        print("3. Both")
        print("4. Exit")
        print()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n--- POWERBALL ---")
            result = generator.generate_powerball()
            print(generator.format_powerball(result))
            print()
            
        elif choice == '2':
            print("\n--- MEGA MILLIONS ---")
            result = generator.generate_mega_millions()
            print(generator.format_mega_millions(result))
            print()
            
        elif choice == '3':
            print("\n--- POWERBALL ---")
            pb_result = generator.generate_powerball()
            print(generator.format_powerball(pb_result))
            print()
            
            print("--- MEGA MILLIONS ---")
            mm_result = generator.generate_mega_millions()
            print(generator.format_mega_millions(mm_result))
            print()
            
        elif choice == '4':
            print("\nThank you for using Lottery Number Generator!")
            print("Good luck!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
