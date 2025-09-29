"""
Simple Math Calculator
"""

import operator
from typing import List, Dict, Union

class Calculator:
    """
    A simple calculator class that supports basic arithmetic operations
    and keeps a history of calculations.
    """

    def __init__(self):
        """Initializes the calculator with an empty history."""
        self.history: List[Dict[str, Union[str, float]]] = []

    def _add_to_history(self, operation: str, num1: float, num2: float, result: float):
        """Adds the performed operation to the calculation history."""
        self.history.append({
            "operation": operation,
            "num1": num1,
            "num2": num2,
            "result": result
        })

    def add(self, num1: float, num2: float) -> float:
        """
        Adds two numbers.

        Args:
            num1: The first number.
            num2: The second number.

        Returns:
            The sum of num1 and num2.
        """
        result = operator.add(num1, num2)
        self._add_to_history("add", num1, num2, result)
        return result

    def subtract(self, num1: float, num2: float) -> float:
        """
        Subtracts the second number from the first.

        Args:
            num1: The number to subtract from.
            num2: The number to subtract.

        Returns:
            The difference between num1 and num2.
        """
        result = operator.sub(num1, num2)
        self._add_to_history("subtract", num1, num2, result)
        return result

    def multiply(self, num1: float, num2: float) -> float:
        """
        Multiplies two numbers.

        Args:
            num1: The first number.
            num2: The second number.

        Returns:
            The product of num1 and num2.
        """
        result = operator.mul(num1, num2)
        self._add_to_history("multiply", num1, num2, result)
        return result

    def divide(self, num1: float, num2: float) -> float:
        """
        Divides the first number by the second.

        Args:
            num1: The dividend.
            num2: The divisor.

        Returns:
            The quotient of num1 and num2.

        Raises:
            ValueError: If the divisor is zero.
        """
        if num2 == 0:
            raise ValueError("Division by zero is not allowed.")
        result = operator.truediv(num1, num2)
        self._add_to_history("divide", num1, num2, result)
        return result

    def power(self, num1: float, num2: float) -> float:
        """
        Raises the first number to the power of the second.

        Args:
            num1: The base.
            num2: The exponent.

        Returns:
            The result of num1 raised to the power of num2.
        """
        result = operator.pow(num1, num2)
        self._add_to_history("power", num1, num2, result)
        return result

    def get_history(self) -> List[Dict[str, Union[str, float]]]:
        """
        Returns the history of calculations.

        Returns:
            A list of dictionaries, where each dictionary represents a calculation.
        """
        return self.history

def main():
    """Main function to run the calculator interactively."""
    calc = Calculator()
    print("Welcome to the Simple Math Calculator!")
    print("Supported operations: add, subtract, multiply, divide, power")
    print("Type 'history' to view calculation history.")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("Enter calculation (e.g., 'add 5 3'): ").strip().lower()
            if user_input == "exit":
                break
            elif user_input == "history":
                if not calc.history:
                    print("No calculations yet.")
                else:
                    print("\n--- Calculation History ---")
                    for entry in calc.history:
                        print(f"  {entry['operation']} {entry['num1']} {entry['num2']} = {entry['result']}")
                    print("-------------------------\n")
                continue

            parts = user_input.split()
            if len(parts) != 3:
                print("Invalid input format. Please use 'operation num1 num2'.")
                continue

            operation = parts[0]
            num1 = float(parts[1])
            num2 = float(parts[2])

            result = None
            if operation == "add":
                result = calc.add(num1, num2)
            elif operation == "subtract":
                result = calc.subtract(num1, num2)
            elif operation == "multiply":
                result = calc.multiply(num1, num2)
            elif operation == "divide":
                result = calc.divide(num1, num2)
            elif operation == "power":
                result = calc.power(num1, num2)
            else:
                print(f"Unknown operation: {operation}")
                continue

            print(f"Result: {result}")

        except ValueError as ve:
            print(f"Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()