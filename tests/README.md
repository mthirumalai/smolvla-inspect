# Tests for smolvla-inspect

This directory contains the test suite for the smolvla-inspect project.

## Running Tests

### Quick Start

From the project root directory:

```bash
# Run all tests
./test.sh

# Or run with pytest directly
python -m pytest tests/ -v
```

### Test Categories

Tests are organized with markers:

- `unit`: Fast unit tests that don't require heavy dependencies
- `integration`: Integration tests that may be slower
- `slow`: Tests that take significant time to run

### Running Specific Test Types

```bash
# Run only fast unit tests
python -m pytest tests/ -v -m "not slow and not integration"

# Run only integration tests
python -m pytest tests/ -v -m "integration"

# Run with coverage report
python -m pytest tests/ --cov=smolvla_inspect --cov-report=term-missing
```

### Dependencies

The test suite requires:

- `pytest>=7.0`
- `pytest-cov>=4.0` (for coverage reports)

Install with: `pip install pytest pytest-cov`

## Test Structure

### `test_scene_models.py`

Tests for the scene understanding and model caching functionality:

- **TestParseTaskObjects**: Task string parsing
- **TestBoxIoU**: Bounding box IoU computation
- **TestPerClassNMS**: Non-maximum suppression algorithm
- **TestCachedSceneModels**: Model caching functionality
- **TestDetectObjectsIntegration**: Object detection with mocking
- **TestSegmentSceneIntegration**: Scene segmentation with mocking
- **TestEndToEndSceneProcessing**: Complete pipeline tests

### Key Features Tested

1. **Model Caching**: Ensures `CachedSceneModels` loads models only once and reuses them
2. **Algorithm Correctness**: Tests core algorithms like IoU, NMS, task parsing
3. **Integration**: Tests the complete detect → segment pipeline
4. **Error Handling**: Tests graceful handling of edge cases

### Mocking Strategy

Tests use mocking to avoid downloading actual ML models:

- OWL-ViT v2 models are mocked via `transformers` patches
- SAM models are mocked via `segment_anything` patches
- PyTorch tensors are mocked to return predictable numpy arrays

This keeps tests fast and doesn't require GPU resources or large model downloads.

## Continuous Integration

Tests run automatically on GitHub Actions for:

- Python 3.11 and 3.12
- Ubuntu latest
- Coverage reporting via Codecov

See `.github/workflows/test.yml` for the full CI configuration.