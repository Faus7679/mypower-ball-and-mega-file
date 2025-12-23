# Lottery Number Generator

A Python application to randomly generate lottery numbers for **Powerball** and **Mega Millions** games.

## Features

- **Powerball Generator**: Generates 5 unique numbers (1-69) + 1 Powerball number (1-26)
- **Mega Millions Generator**: Generates 5 unique numbers (1-70) + 1 Mega Ball number (1-25)
- Interactive command-line interface
- Option to generate numbers for both games at once

## Requirements

- Python 3.6 or higher

## Usage

Run the application:

```bash
python3 lottery_generator.py
```

Or make it executable and run directly:

```bash
chmod +x lottery_generator.py
./lottery_generator.py
```

### Menu Options

1. **Powerball** - Generate Powerball numbers
2. **Mega Millions** - Generate Mega Millions numbers
3. **Both** - Generate numbers for both games
4. **Exit** - Quit the application

## Example Output

```
============================================================
              Lottery Number Generator
============================================================

Select a game:
1. Powerball
2. Mega Millions
3. Both
4. Exit

Enter your choice (1-4): 3

--- POWERBALL ---
Numbers: 5 - 12 - 23 - 45 - 67 | Powerball: 18

--- MEGA MILLIONS ---
Numbers: 8 - 15 - 31 - 52 - 69 | Mega Ball: 22
```

## Game Rules

### Powerball
- Choose 5 numbers from 1 to 69
- Choose 1 Powerball number from 1 to 26

### Mega Millions
- Choose 5 numbers from 1 to 70
- Choose 1 Mega Ball number from 1 to 25

## License

Apache License 2.0
