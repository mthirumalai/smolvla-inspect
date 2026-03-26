#!/bin/bash
# Test runner script for smolvla-inspect

set -e  # Exit on any error

echo "Running smolvla-inspect test suite..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/ -v

# Run tests with coverage
echo "Running tests with coverage..."
python -m pytest tests/ --cov=smolvla_inspect --cov-report=term-missing

# Run only fast tests (excluding integration tests)
echo "Running fast tests only..."
python -m pytest tests/ -v -m "not slow and not integration"

echo "All tests completed successfully!"