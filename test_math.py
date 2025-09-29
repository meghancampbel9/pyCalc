"""
Unit tests for the simple math calculator module.
"""
import pytest
import sys
import io
from unittest.mock import patch
from math import Calculator, format_result, validate_number


class TestCalculator:
    """Test cases for the Calculator class."""
    
    def setup_method(self):
        """Set up a fresh calculator instance for each test."""
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        """Test adding positive numbers."""
        result = self.calc.add(2, 3)
        assert result == 5
        assert len(self.calc.get_history()) == 1
    
    def test_add_negative_numbers(self):
        """Test adding negative numbers."""
        result = self.calc.add(-2, -3)
        assert result == -5
        assert len(self.calc.get_history()) == 1
    
    def test_add_mixed_numbers(self):
        """Test adding mixed positive and negative numbers."""
        result = self.calc.add(5, -3)
        assert result == 2
        assert len(self.calc.get_history()) == 1
    
    def test_add_float_numbers(self):
        """Test adding floating point numbers."""
        result = self.calc.add(2.5, 3.7)
        assert abs(result - 6.2) < 0.0001
        assert len(self.calc.get_history()) == 1
    
    def test_subtract_positive_numbers(self):
        """Test subtracting positive numbers."""
        result = self.calc.subtract(5, 3)
        assert result == 2
        assert len(self.calc.get_history()) == 1
    
    def test_subtract_negative_numbers(self):
        """Test subtracting negative numbers."""
        result = self.calc.subtract(-2, -3)
        assert result == 1
        assert len(self.calc.get_history()) == 1
    
    def test_subtract_mixed_numbers(self):
        """Test subtracting mixed positive and negative numbers."""
        result = self.calc.subtract(5, -3)
        assert result == 8
        assert len(self.calc.get_history()) == 1
    
    def test_multiply_positive_numbers(self):
        """Test multiplying positive numbers."""
        result = self.calc.multiply(3, 4)
        assert result == 12
        assert len(self.calc.get_history()) == 1
    
    def test_multiply_negative_numbers(self):
        """Test multiplying negative numbers."""
        result = self.calc.multiply(-3, -4)
        assert result == 12
        assert len(self.calc.get_history()) == 1
    
    def test_multiply_mixed_numbers(self):
        """Test multiplying mixed positive and negative numbers."""
        result = self.calc.multiply(-3, 4)
        assert result == -12
        assert len(self.calc.get_history()) == 1
    
    def test_multiply_by_zero(self):
        """Test multiplying by zero."""
        result = self.calc.multiply(5, 0)
        assert result == 0
        assert len(self.calc.get_history()) == 1
    
    def test_divide_positive_numbers(self):
        """Test dividing positive numbers."""
        result = self.calc.divide(12, 3)
        assert result == 4.0
        assert len(self.calc.get_history()) == 1
    
    def test_divide_negative_numbers(self):
        """Test dividing negative numbers."""
        result = self.calc.divide(-12, -3)
        assert result == 4.0
        assert len(self.calc.get_history()) == 1
    
    def test_divide_mixed_numbers(self):
        """Test dividing mixed positive and negative numbers."""
        result = self.calc.divide(-12, 3)
        assert result == -4.0
        assert len(self.calc.get_history()) == 1
    
    def test_divide_by_zero(self):
        """Test dividing by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(5, 0)
        assert len(self.calc.get_history()) == 0
    
    def test_power_positive_numbers(self):
        """Test raising positive numbers to powers."""
        result = self.calc.power(2, 3)
        assert result == 8
        assert len(self.calc.get_history()) == 1
    
    def test_power_negative_base(self):
        """Test raising negative base to positive power."""
        result = self.calc.power(-2, 3)
        assert result == -8
        assert len(self.calc.get_history()) == 1
    
    def test_power_zero_exponent(self):
        """Test raising number to power of zero."""
        result = self.calc.power(5, 0)
        assert result == 1
        assert len(self.calc.get_history()) == 1
    
    def test_power_fractional_result(self):
        """Test power operations that result in fractions."""
        result = self.calc.power(4, 0.5)
        assert abs(result - 2.0) < 0.0001
        assert len(self.calc.get_history()) == 1
    
    def test_history_start_empty(self):
        """Test that calculation history starts empty."""
        assert len(self.calc.get_history()) == 0
    
    def test_history_accumulation(self):
        """Test that calculations are added to history."""
        self.calc.add(1, 2)
        self.calc.subtract(5, 3)
        self.calc.multiply(4, 6)
        
        history = self.calc.get_history()
        assert len(history) == 3
        assert "1 + 2 = 3" in history
        assert "5 - 3 = 2" in history
        assert "4 * 6 = 24" in history
    
    def test_clear_history(self):
        """Test clearing calculation history."""
        self.calc.add(1, 2)
        self.calc.subtract(5, 3)
        assert len(self.calc.get_history()) == 2
        
        self.calc.clear_history()
        assert len(self.calc.get_history()) == 0
    
    def test_history_is_copy(self):
        """Test that get_history returns a copy, not the original list."""
        self.calc.add(1, 2)
        history = self.calc.get_history()
        # Modify the returned list
        history.append("fake calculation")
        # Original history should be unchanged
        assert len(self.calc.get_history()) == 1
        assert len(history) == 2

class TestFormatResult:
    """Test cases for the format_result function."""
    
    def test_format_addition(self):
        """Test formatting addition results."""
        result = format_result("Addition", 5, 3, 8)
        assert result == "Addition: 5 and 3 = 8"
    
    def test_format_subtraction(self):
        """Test formatting subtraction results."""
        result = format_result("Subtraction", 10, 7, 3)
        assert result == "Subtraction: 10 and 7 = 3"
    
    def test_format_with_floats(self):
        """Test formatting with floating point numbers."""
        result = format_result("Division", 15, 4, 3.75)
        assert result == "Division: 15 and 4 = 3.75"


class TestValidateNumber:
    """Test cases for the validate_number function."""
    
    def test_validate_positive_integer(self):
        """Test validating positive integers."""
        assert validate_number(42) == True
        assert validate_number("42") == True
    
    def test_validate_negative_integer(self):
        """Test validating negative integers."""
        assert validate_number(-42) == True
        assert validate_number("-42") == True
    
    def test_validate_positive_float(self):
        """Test validating positive floats."""
        assert validate_number(3.14) == True
        assert validate_number("3.14") == True
    
    def test_validate_negative_float(self):
        """Test validating negative floats."""
        assert validate_number(-3.14) == True
        assert validate_number("-3.14") == True
    
    def test_validate_zero(self):
        """Test validating zero."""
        assert validate_number(0) == True
        assert validate_number("0") == True
        assert validate_number(0.0) == True
        assert validate_number("0.0") == True
    
    def test_validate_invalid_strings(self):
        """Test validating invalid string inputs."""
        assert validate_number("abc") == False
        assert validate_number("") == False
        assert validate_number("hello world") == False
        assert validate_number("123abc") == False
    
    def test_validate_none(self):
        """Test validating None input."""
        assert validate_number(None) == False
    
    def test_validate_list(self):
        """Test validating list input."""
        assert validate_number([1, 2, 3]) == False


class TestCalculatorIntegration:
    """Integration tests for the Calculator class."""
    
    def test_multiple_operations_chain(self):
        """Test chaining multiple operations."""
        calc = Calculator()
        
        result1 = calc.add(10, 5)  # 15
        result2 = calc.subtract(result1, 3)  # 12
        result3 = calc.multiply(result2, 2)  # 24
        result4 = calc.divide(result3, 4)  # 6
        
        assert result4 == 6
        assert len(calc.get_history()) == 4
    
    def test_calculator_performance(self):
        """Test calculator with many operations."""
        calc = Calculator()
        
        for i in range(100):
            calc.add(i, i)
        
        assert len(calc.get_history()) == 100
        assert calc.get_history()[0] == "0 + 0 = 0"
        assert calc.get_history()[-1] == "99 + 99 = 198"


class TestEdgeCases:
    """Test cases for edge cases and boundary conditions."""
    
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        calc = Calculator()
        result = calc.add(999999999, 1)
        assert result == 1000000000
    
    def test_very_small_numbers(self):
        """Test with very small decimal numbers."""
        calc = Calculator()
        result = calc.add(0.000001, 0.000002)
        assert abs(result - 0.000003) < 1e-10
    
    def test_result_preserves_type(self):
        """Test that operations preserve numeric types appropriately."""
        calc = Calculator()
        
        # Adding two ints should return int
        int_result = calc.add(2, 3)
        assert isinstance(int_result, int)
        
        # Adding two floats should return float
        float_result = calc.add(2.0, 3.0)
        assert isinstance(float_result, float)
        
        # Mixing int and float should return float
        mixed_result = calc.add(2, 3.0)
        assert isinstance(mixed_result, float)


# Pytest fixtures for common test setup
@pytest.fixture
def calculator():
    """Fixture that provides a Calculator instance."""
    return Calculator()


@pytest.fixture
def calculator_with_history():
    """Fixture that provides a Calculator instance with some history."""
    calc = Calculator()
    calc.add(1, 2)
    calc.subtract(5, 3)
    calc.multiply(4, 6)
    return calc


# Parametrized tests for more comprehensive coverage
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 5, 5),
    (-1, 1, 0),
    (10.5, 2.5, 13.0),
    (-5.5, -2.5, -8.0),
])
def test_add_parametrized(a, b, expected, calculator):
    """Parametrized test for addition operation."""
    result = calculator.add(a, b)
    assert result == expected


@pytest.mark.parametrize("a,b,expected", [
    (5, 2, 3),
    (10, 10, 0),
    (-3, 1, -4),
    (7.5, 2.5, 5.0),
])
def test_subtract_parametrized(a: int, b: int, expected: int, calculator: Calculator) -> None:
    """Parametrized test for subtraction operation."""
    result = calculator.subtract(a, b)
    assert result == expected


@pytest.mark.parametrize("value", [
    1, "1", 1.0, "1.0", -5, "-5", 3.14, "3.14", 0, "0"
])
def test_validate_number_parametrized(value):
    """Parametrized test for number validation."""
    assert validate_number(value) == True


@pytest.mark.parametrize("value", [
    "abc", "", None, [], [1, 2, 3], "hello", "123abc"
])
def test_validate_number_invalid_parametrized(value):
    """Parametrized test for invalid number validation."""
    assert validate_number(value) == False


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
