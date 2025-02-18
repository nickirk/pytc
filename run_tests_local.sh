#!/bin/bash
# Local test runner that mimics GitHub Actions workflow

echo "Running tests locally to simulate GitHub Actions..."

# Install dependencies without --user flag
echo "Installing dependencies..."
python -m pip install coverage

# List and verify all test files
echo "Looking for test files in:"
TEST_FILES=$(find pytc -type f -name "test_*.py")
echo "$TEST_FILES"

# Run tests with explicit test file list
echo "Running tests with coverage..."
TEST_PATTERN=$(echo "$TEST_FILES" | tr '\n' ' ')
python -m coverage run -m unittest $TEST_PATTERN -v

# Generate coverage report
echo "Generating coverage report..."
python -m coverage xml
python -m coverage report --include="pytc/*" --omit="pytc/*/test*"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Coverage report generated for:"
    find pytc -name "*.py" -not -path "*/test*" | sort
else
    echo "❌ Tests failed"
    exit 1
fi
