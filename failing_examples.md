# 🚨 Testing Pipeline Failures

This file contains examples of how to intentionally fail the CI/CD pipeline to test your GitHub App's failure handling capabilities.

## Quick Failure Methods

### 1. Break Basic Operation Test

Edit `test_calculator.py` line ~26:

```python
# Change this line:
assert value == 5
# To this:
assert value == 5
```

### 2. Break Division Test

Edit `test_calculator.py` around line ~107:

```python
def test_divide_by_zero(self):
    """Test dividing by zero raises ValueError."""
    # Change this to expect no error (will fail):
    result = self.calc.divide(5, 0)
    assert result == 0  # This will fail
```

### 3. Break Syntax

Add this line anywhere in `calculator.py`:

```python
def broken_syntax_function(:  # Missing closing parenthesis
```

### 4. Break Import Statement

Add this line at the top of `test_calculator.py`:

```python
import non_existent_module_for_failure_testing
```

### 5. Break Logic Flow

Edit `test_calculator.py` line ~78:

```python
def test_multiply_by_zero(self):
    """Test multiplying by zero."""
    result = self.calc.multiply(5, 0)
    assert result == 999  # Wrong expected value
    assert len(self.calc.get_history()) == 1
```

## Comprehensive Failure Examples

Create a branch called `test-failure` and try these out:

```bash
# Create failure branch
git checkout -b test-failure

# Make a commit that fails
git add .
git commit -m "test: intentionally break test for GitHub App testing"
git push origin test-failure

# Create PR to trigger pipeline with failure
# Then merge/discard and try next failure type
```

## Different Types of Failures to Test

1. **Assertion Failures** - Wrong expected values
2. **Exception Handling** - Code that raises unexpected errors
3. **Import Failures** - Missing modules or dependency issues
4. **Syntax Errors** - Malformed Python code
5. **Type Errors** - Type-related failures
6. **Logic Errors** - Algorithmic mistakes
7. **Resource Errors** - Memory or performance issues

## Pipeline Trigger Commands

```bash
# Quick commit that will pass
echo "# $(date)" >> README.md
git add README.md
git commit -m "test: trigger passing pipeline"
git push origin main

# Quick commit that will fail
echo "assert False" >> test_calculator.py
git add test_calculator.py
git commit -m "test: trigger failing pipeline"
git push origin main

# Revert the failure
git checkout HEAD~1 test_calculator.py
git commit -m "test: revert failure"
git push origin main
```

## Testing Workflow

1. **Stage 1**: Run passing tests to establish baseline
2. **Stage 2**: Introduce different failure types systematically
3. **Stage 3**: Test your GitHub App's responses to each failure type
4. **Stage 4**: Verify recovery mechanisms work correctly
5. **Stage 5**: Test edge cases and complex scenarios

This systematic approach will help you thoroughly test all aspects of your GitHub App's CI/CD monitoring capabilities.
