# Python Math Calculator

A Python calculator application designed to test GitHub CI/CD workflows and GitHub Apps. This project includes comprehensive unit tests and automated CI/CD pipelines.

## 🚀 Quick Start

### Installation

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

#### Command Line Interface

```bash
# Run the calculator
python calculator.py

# Or using the entry point
simple-math
```

The calculator supports basic operations: `+`, `-`, `*`, `/`, `^`

#### Python Module

```python
from calculator import Calculator

calc = Calculator()
result = calc.add(5, 3)  # 8
result = calc.multiply(4, 6)  # 24
result = calc.divide(10, 2)  # 5.0

# View calculation history
print(calc.get_history())
```

## 🧪 Testing

### Run Tests Locally

```bash
# Run all tests
pytest test_calculator.py -v

# Run with coverage
pytest test_calculator.py --cov=calculator --cov-report=html

# View coverage report
open htmlcov/index.html 
```

### Test Components

The test suite includes:

1. **Basic Operations Tests** - Addition, subtraction, multiplication, division, power
2. **Edge Case Tests** - Division by zero, large numbers, floating point precision
3. **Integration Tests** - Multi-step calculations, history tracking
4. **Type Validation Tests** - Input validation and error handling
5. **Performance Tests** - Large batch operations

### Test Categories

- Unit tests for individual methods
- Integration tests for workflow scenarios
- Parametrized tests for comprehensive input coverage
- Fixture-based tests for consistent setup
- Edge case and boundary condition tests

## 🔄 CI/CD Pipeline

This project includes a GitHub Actions workflow that runs on every push and pull request.

### Pipeline Stages

1. **Matrix Testing** - Tests across Python versions 3.8-3.12
2. **Code Quality** - Flake8 linting, Black formatting, MyPy type checking
3. **Security Scanning** - Safety and Bandit security analysis
4. **Coverage Reporting** - Code coverage metrics with HTML reports
5. **Package Building** - Wheel and source distribution validation

### Trigger Conditions

The pipeline runs on:
- Push to `main` or `develop` branches
- Pull requests targeting `main`
- Manual workflow dispatch

### Pipeline Features

- **Parallel Job Execution** - Multiple Python versions tested simultaneously
- **Artifact Generation** - Coverage reports and security scans saved as artifacts
- **PR Comments** - Automatic test result comments on pull requests
- **Build Validation** - Package building and twine validation
- **Security Integration** - Automated security vulnerability scanning

## 🚨 Testing Pipeline Failures

To test your GitHub App's ability to handle pipeline failures, you can intentionally break tests:

### Method 1: Break Test Assertions

Edit `test_calculator.py` and modify any `assert` statement:

```python
# Change this:
assert result == 5

# To this (will fail):
assert result == 50
```

### Method 2: Break Code Syntax

Add a syntax error to `calculator.py`:

```python
# Add this line anywhere in calculator.py
def broken_function(:
```

### Method 3: Break Import Dependencies

Add an invalid import in the test file:

```python
# Add this at the top of test_calculator.py
import nonexistent_module
```

### Method 4: Break Conditional Logic

Modify a test condition:

```python
# Change a test like this:
def test_add_positive_numbers(self):
    result = self.calc.add(2, 3)
    assert result == 5  # This will fail if you change 5 to something else
```

## 📁 Project Structure

```
pyLaunch/
├── .github/
│   └── workflows/
│       └── test.yml              # CI/CD pipeline configuration
├── .gitignore                    # Git ignore patterns
├── pytest.ini                   # Pytest configuration
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup configuration
├── calculator.py        # Main calculator module
└── test_calculator.py   # Comprehensive test suite
```

## 📊 Test Coverage

The test suite provides comprehensive coverage including:

- **Basic Operations**: All mathematical operations
- **Error Handling**: Division by zero, invalid inputs
- **Data Validation**: Type checking and input validation
- **History Management**: Calculation history and memory
- **Edge Cases**: Large numbers, small decimals, boundary conditions
- **Integration Scenarios**: Complex multi-step calculations

## 🔧 Configuration Files

### pytest.ini
Configures pytest with coverage reporting and test discovery patterns.

### requirements.txt
Development dependencies including testing and code quality tools.

### setup.py
Package configuration with console entry points and metadata.

### .github/workflows/test.yml
Complete CI/CD pipeline configuration with multiple stages and Python version matrix.

## 🛠️ Development Tools

The project uses several development tools:

- **pytest** - Test runner and framework
- **flake8** - Code linting and style checking
- **black** - Code formatting
- **mypy** - Static type checking
- **coverage** - Code coverage analysis
- **bandit** - Security vulnerability scanning
- **safety** - Dependency vulnerability checking

## 📝 Usage Examples

### Example Test Runs

```bash
# Run specific test class
pytest test_calculator.py::TestCalculator -v

# Run with coverage
pytest --cov=calculator --cov-report=term-missing

# Run parametrized tests only
pytest test_calculator.py -k "parametrized" -v

# Run fast tests (skip slow ones)
pytest test_calculator.py -m "not slow" -v
```

### Example CI Pipeline Trigger

Create a commit that will trigger the pipeline:

```bash
# Make sure you're on main branch
git add .
git commit -m "Add comprehensive test suite and CI pipeline"
git push origin main
```

## 🚀 GitHub App Testing Scenarios

This project is designed to test various aspects of GitHub Apps:

1. **Monitoring CI Status** - Track pipeline success/failure states
2. **Handling Failures** - Test app behavior on failed workflows
3. **Artifact Processing** - Test app's ability to access and process workflow artifacts
4. **Webhook Events** - Test app responses to various GitHub events
5. **API Integration** - Test GitHub API usage for status updates and comments

## 📈 Monitoring and Metrics

The CI pipeline provides:

- Test execution time and results
- Code coverage percentages
- Security vulnerability reports
- Build artifact validation
- Performance metrics for large test suites

## 🔍 Troubleshooting

### Common Issues

1. **Test Failures**: Check for syntax errors or broken assertions
2. **Import Errors**: Verify all dependencies are installed
3. **Coverage Issues**: Ensure test_*.py files are properly named
4. **CI Failures**: Check workflow YAML syntax and step dependencies

### Getting Help

- Review the GitHub Actions logs for detailed error messages
- Check test output locally before pushing
- Validate YAML syntax in workflow files
- Ensure all dependencies are listed in requirements.txt

## 📄 License

This project is provided as-is for testing GitHub Apps and CI/CD workflows.
