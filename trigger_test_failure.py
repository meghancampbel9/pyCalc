#!/usr/bin/env python3
"""
Script to trigger test failures for GitHub App testing.
This allows easy testing of CI/CD pipeline failure scenarios.
"""

import os
import sys
from pathlib import Path


def create_syntax_error():
    """Create a syntax error in the main module."""
    file_path = Path("calculator.py")
    with open(file_path, "a") as f:
        f.write("\n# Intentional syntax error for testing\n")
        f.write("def broken_function(:\n")  # Missing closing parenthesis
        f.write("    pass\n")


def break_test_assertion():
    """Break a test assertion."""
    file_path = Path("test_calculator.py")

    # Read current content
    with open(file_path, "r") as f:
        content = f.read()

        # Replace assertion
        content = content.replace(
            "assert result == 8\n",
            "assert result == 888  # Intentionally wrong for testing\n",
        )

    # Write back with modification
    with open(file_path, "w") as f:
        f.write(content)


def add_invalid_import():
    """Add an invalid import to test import failures."""
    file_path = Path("test_calculator.py")

    with open(file_path, "r") as f:
        content = f.read()

    # Add invalid import after existing imports
    content = content.replace(
        "from math import Calculator, format_result, validate_number\n",
        "from math import Calculator, format_result, validate_number\n"
        "import none_existent_module_for_testing  # Will cause ImportError\n",
    )

    with open(file_path, "w") as f:
        f.write(content)


def revert_changes():
    """Revert all changes by checking out from git."""
    import subprocess

    try:
        subprocess.run(
            ["git", "checkout", "--", "calculator.py", "test_calculator.py"], check=True
        )
        print("✅ Changes reverted successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to revert changes")


def main():
    """Main function to orchestrate test failure scenarios."""
    print("🚨 Test Failure Trigger Script")
    print("=" * 40)
    print("1. Create syntax error")
    print("2. Break test assertion")
    print("3. Add invalid import")
    print("4. Revert all changes")
    print("5. Exit")

    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()

            if choice == "1":
                create_syntax_error()
                print("✅ Syntax error created - next commit will fail")

            elif choice == "2":
                break_test_assertion()
                print("✅ Test assertion broken - next commit will fail")

            elif choice == "3":
                add_invalid_import()
                print("✅ Invalid import added - next commit will fail")

            elif choice == "4":
                revert_changes()

            elif choice == "5":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
