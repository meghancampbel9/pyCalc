"""
Simple Math Calculator
"""

import operator
from typing import List, Dict, Any, Optional

class Calculator:
    """
    A simple calculator class that supports basic arithmetic operations
    and keeps a history of calculations.
    """

    def __init__(self):
        """
        Initializes the Calculator with an empty history.
        """
        self.history: List[Dict[str, Any]] = []

    def _log_operation(self, operation: str, num1: float, num2: float, result: float) -> None:
        """Logs the details of an operation to the history."""
        self.history.append({
            "operation": operation,
            "operands": [num1, num2],
            "result": result
        })

    def add(self, num1: float, num2: float) -> float:
        """
        Adds two numbers and logs the operation.

        Args:
            num1: The first number.
            num2: The second number.

        Returns:
            The sum of num1 and num2.
        """
        result = num1 + num2
        self._log_operation("add", num1, num2, result)
        return result

    def subtract(self, num1: float, num2: float) -> float:
        """
        Subtracts the second number from the first and logs the operation.

        Args:
            num1: The number to subtract from.
            num2: The number to subtract.

        Returns:
            The difference between num1 and num2.
        """
        result = num1 - num2
        self._log_operation("subtract", num1, num2, result)
        return result

    def multiply(self, num1: float, num2: float) -> float:
        """
        Multiplies two numbers and logs the operation.

        Args:
            num1: The first number.
            num2: The second number.

        Returns:
            The product of num1 and num2.
        """
        result = num1 * num2
        self._log_operation("multiply", num1, num2, result)
        return result

    def divide(self, num1: float, num2: float) -> float:
        """
        Divides the first number by the second and logs the operation.

        Args:
            num1: The dividend.
            num2: The divisor.

        Returns:
            The quotient of num1 and num2.

        Raises:
            ValueError: If the divisor is zero.
        """
        if num2 == 0:
            raise ValueError("Cannot divide by zero.")
        result = num1 / num2
        self._log_operation("divide", num1, num2, result)
        return result

    def power(self, base: float, exponent: float) -> float:
        """
        Calculates the base raised to the power of the exponent and logs the operation.

        Args:
            base: The base number.
            exponent: The exponent.

        Returns:
            The result of base raised to the power of exponent.
        """
        result = base ** exponent
        self._log_operation("power", base, exponent, result)
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of all calculations performed.

        Returns:
            A list of dictionaries, where each dictionary represents a calculation.
        """
        return self.history

    def clear_history(self) -> None:
        """Clears the calculation history."""
        self.history = []

def main() -> None:
    """Main function to run the calculator interactively."""
    calc = Calculator()
    print("Welcome to the Simple Math Calculator!")
    print("Supported operations: +, -, *, /, ^")
    print("Type 'history' to view calculation history, 'clear' to clear it, or 'quit' to exit.")

    while True:
        try:
            user_input = input("Enter calculation (e.g., 5 + 3): ").strip()
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'history':
                if not calc.get_history():
                    print("History is empty.")
                else:
                    print("\n--- Calculation History ---")
                    for entry in calc.get_history():
                        print(f"  {entry['operation']}({entry['operands'][0]}, {entry['operands'][1]}) = {entry['result']}")
                    print("-------------------------\n")
                continue
            elif user_input.lower() == 'clear':
                calc.clear_history()
                print("History cleared.")
                continue

            parts = user_input.split()
            if len(parts) != 3:
                print("Invalid input format. Please use 'number operator number'.")
                continue

            num1_str, operator_str, num2_str = parts

            try:
                num1 = float(num1_str)
                num2 = float(num2_str)
            except ValueError:
                print("Invalid numbers. Please enter valid numeric values.")
                continue

            result: Optional[float] = None
            operation_name: Optional[str] = None

            if operator_str == '+':
                result = calc.add(num1, num2)
                operation_name = "add"
            elif operator_str == '-':
                result = calc.subtract(num1, num2)
                operation_name = "subtract"
            elif operator_str == '*':
                result = calc.multiply(num1, num2)
                operation_name = "multiply"
            elif operator_str == '/':
                try:
                    result = calc.divide(num1, num2)
                    operation_name = "divide"
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
            elif operator_str == '^':
                result = calc.power(num1, num2)
                operation_name = "power"
            else:
                print(f"Unknown operator: {operator_str}. Supported operators are +, -, *, /, ^.")
                continue

            print(f"Result: {result}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()