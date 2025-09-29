"""
Simple Math Calculator Module

This module provides basic mathematical operations for testing purposes.
"""
import logging
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Calculator:
    """A simple calculator class for basic mathematical operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        logger.info(f"Added {a} and {b}, result: {result}")
        return result
    
    def subtract(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        logger.info(f"Subtracted {b} from {a}, result: {result}")
        return result
    
    def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        logger.info(f"Multiplied {a} and {b}, result: {result}")
        return result
    
    def divide(self, a: Union[int, float], b: Union[int, float]) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        logger.info(f"Divided {a} by {b}, result: {result}")
        return result
    
    def power(self, a: Union[int, float], b: Union[int, float]) -> float:
        """Raise a number to a power."""
        result = a ** b
        self.history.append(f"{a} ^ {b} = {result}")
        logger.info(f"Raised {a} to power {b}, result: {result}")
        return result
    
    def get_history(self) -> list:
        """Get calculation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()
        logger.info("Calculation history cleared")


def format_result(operation: str, a: Union[int, float], b: Union[int, float], result: Union[int, float]) -> str:
    """Format a calculation result for display."""
    return f"{operation}: {a} and {b} = {result}"


def validate_number(value: Union[int, float, str]) -> bool:
    """Validate if a value can be converted to a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def main():
    """Main function for command-line usage."""
    calc = Calculator()
    print("Simple Calculator")
    print("Operations: +, -, *, /, ^")
    print("Type 'quit' to exit")
    
    while True:
        try:
            print("\nEnter an expression (e.g., '5 + 3') or 'history' to see calculations:")
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'history':
                history = calc.get_history()
                if history:
                    print("Calculation History:")
                    for entry in history:
                        print(f"  {entry}")
                else:
                    print("No calculations performed yet.")
                continue
            
            parts = user_input.split()
            if len(parts) != 3:
                print("Please enter in format: 'number operator number'")
                continue
            
            a, operator, b = parts
            
            if not validate_number(a) or not validate_number(b):
                print("Please enter valid numbers")
                continue
            
            a, b = float(a), float(b)
            
            if operator == '+':
                result = calc.add(a, b)
            elif operator == '-':
                result = calc.subtract(a, b)
            elif operator == '*':
                result = calc.multiply(a, b)
            elif operator == '/':
                result = calc.divide(a, b)
            elif operator == '^':
                result = calc.power(a, b)
            else:
                print("Invalid operator. Use +, -, *, /, or ^")
                continue
            
            print(f"Result: {result}")
            
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
