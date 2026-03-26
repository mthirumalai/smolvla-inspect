"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from smolvla_inspect.diagnostic.models import DetectedObject


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Create sample detection objects for testing."""
    return [
        DetectedObject(
            label="cup",
            box=(10, 10, 50, 50),
            score=0.8,
            mask=None
        ),
        DetectedObject(
            label="bowl",
            box=(60, 60, 100, 100),
            score=0.9,
            mask=None
        )
    ]


@pytest.fixture
def mock_transformers_models(monkeypatch):
    """Mock transformers models to avoid downloading during tests."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    def mock_processor_from_pretrained(*args, **kwargs):
        return mock_processor

    def mock_model_from_pretrained(*args, **kwargs):
        mock_model.to.return_value = mock_model
        return mock_model

    monkeypatch.setattr(
        'smolvla_inspect.diagnostic.scene.Owlv2Processor.from_pretrained',
        mock_processor_from_pretrained
    )
    monkeypatch.setattr(
        'smolvla_inspect.diagnostic.scene.Owlv2ForObjectDetection.from_pretrained',
        mock_model_from_pretrained
    )

    return mock_processor, mock_model


@pytest.fixture
def mock_sam_models(monkeypatch):
    """Mock SAM models to avoid downloading during tests."""
    mock_sam = MagicMock()
    mock_predictor = MagicMock()

    def mock_sam_registry(model_type):
        def load_checkpoint(checkpoint):
            mock_sam.to.return_value = mock_sam
            return mock_sam
        return load_checkpoint

    def mock_sam_predictor(sam_model):
        return mock_predictor

    def mock_ensure_checkpoint():
        return "/fake/checkpoint/path"

    monkeypatch.setattr(
        'smolvla_inspect.diagnostic.scene.sam_model_registry',
        {"vit_b": mock_sam_registry}
    )
    monkeypatch.setattr(
        'smolvla_inspect.diagnostic.scene.SamPredictor',
        mock_sam_predictor
    )
    monkeypatch.setattr(
        'smolvla_inspect.diagnostic.scene._ensure_sam_checkpoint',
        mock_ensure_checkpoint
    )

    return mock_sam, mock_predictor