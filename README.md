# Lottery Number Generator

A simple Python application to randomly generate lottery numbers for **Powerball** and **Mega Millions** games.

## Features

- 🎱 **Powerball**: Generates 5 unique numbers (1-69) and 1 Powerball number (1-26)
- 💰 **Mega Millions**: Generates 5 unique numbers (1-70) and 1 Mega Ball number (1-25)
- 🎲 Generate numbers for both games at once
- 🎨 Clean, easy-to-read output format

## Requirements

- Python 3.x (no external dependencies required)

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

1. **Powerball** - Generate Powerball numbers only
2. **Mega Millions** - Generate Mega Millions numbers only
3. **Both Games** - Generate numbers for both games
4. **Exit** - Exit the application

## Example Output

```
==================================================
           POWERBALL NUMBERS
==================================================
Main Numbers: 6 - 13 - 29 - 54 - 55
Powerball:    16
==================================================
```

## Game Rules

### Powerball
- Choose 5 numbers from 1 to 69
- Choose 1 Powerball number from 1 to 26

### Mega Millions
- Choose 5 numbers from 1 to 70
- Choose 1 Mega Ball number from 1 to 25

## License

See LICENSE file for details.

---

**Disclaimer**: This is a random number generator for entertainment purposes only. It does not guarantee winning numbers.
