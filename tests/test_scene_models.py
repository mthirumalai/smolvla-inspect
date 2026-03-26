"""Test suite for scene understanding models and caching functionality."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from smolvla_inspect.diagnostic.scene import (
    CachedSceneModels,
    detect_objects,
    segment_scene,
    parse_task_objects,
    _per_class_nms,
    _box_iou,
)
from smolvla_inspect.diagnostic.models import DetectedObject, SceneSegmentation


class TestParseTaskObjects:
    """Test task object parsing functionality."""

    def test_basic_parsing(self):
        """Test basic task object parsing."""
        result = parse_task_objects("pick up the blue cup and place it in the red bowl")
        expected = ["blue cup", "red bowl", "robot gripper"]
        assert result == expected

    def test_complex_task_parsing(self):
        """Test parsing of more complex task descriptions."""
        result = parse_task_objects("move the small white lego block to the stainless steel container")
        # Should extract meaningful object descriptions
        assert "robot gripper" in result
        assert len(result) >= 2  # Should find at least the objects mentioned

    def test_empty_task_parsing(self):
        """Test parsing empty or minimal task strings."""
        result = parse_task_objects("")
        assert result == ["robot gripper"]  # Should always include robot gripper

    def test_duplicate_removal(self):
        """Test that duplicate objects are removed."""
        result = parse_task_objects("pick up the cup and move the cup to the table")
        # Should not have duplicate "cup"
        cup_count = sum(1 for obj in result if "cup" in obj)
        assert cup_count <= 1


class TestBoxIoU:
    """Test bounding box IoU computation."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes."""
        box = (10, 10, 50, 50)
        iou = _box_iou(box, box)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        iou = _box_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = (0, 0, 20, 20)  # Area = 400
        box2 = (10, 10, 30, 30)  # Area = 400, intersection = 100
        iou = _box_iou(box1, box2)
        expected_iou = 100 / (400 + 400 - 100)  # intersection / union
        assert abs(iou - expected_iou) < 1e-6


class TestPerClassNMS:
    """Test per-class non-maximum suppression."""

    def test_nms_removes_overlapping_detections(self):
        """Test that NMS removes overlapping detections of the same class."""
        # Create overlapping detections of the same class
        detections = [
            DetectedObject(label="cup", box=(10, 10, 50, 50), score=0.9, mask=None),
            DetectedObject(label="cup", box=(15, 15, 55, 55), score=0.7, mask=None),  # Overlapping
            DetectedObject(label="bowl", box=(100, 100, 150, 150), score=0.8, mask=None),
        ]

        result = _per_class_nms(detections, iou_threshold=0.5, max_per_class=3)

        # Should keep the higher scoring "cup" detection and the "bowl" detection
        assert len(result) == 2
        cup_detections = [d for d in result if d.label == "cup"]
        assert len(cup_detections) == 1
        assert cup_detections[0].score == 0.9  # Higher scoring one kept

    def test_nms_respects_max_per_class(self):
        """Test that NMS respects the max_per_class limit."""
        # Create multiple non-overlapping detections of the same class
        detections = [
            DetectedObject(label="cup", box=(10, 10, 20, 20), score=0.9, mask=None),
            DetectedObject(label="cup", box=(30, 30, 40, 40), score=0.8, mask=None),
            DetectedObject(label="cup", box=(50, 50, 60, 60), score=0.7, mask=None),
        ]

        result = _per_class_nms(detections, iou_threshold=0.5, max_per_class=2)

        # Should keep only top 2 detections
        assert len(result) == 2
        assert all(d.label == "cup" for d in result)
        assert result[0].score >= result[1].score  # Should be sorted by score


class TestCachedSceneModels:
    """Test the CachedSceneModels class."""

    def test_initialization(self):
        """Test CachedSceneModels initialization."""
        cache = CachedSceneModels(device="cpu")
        assert cache.device == "cpu"
        assert cache.owl_processor is None
        assert cache.owl_model is None
        assert cache.sam_predictor is None
        assert not cache._owl_loaded
        assert not cache._sam_loaded

    @patch('transformers.Owlv2Processor')
    @patch('transformers.Owlv2ForObjectDetection')
    def test_get_owl_loads_once(self, mock_model_class, mock_processor_class):
        """Test that OWL-ViT models are loaded only once."""
        # Mock the model loading
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        cache = CachedSceneModels(device="cpu")

        # First call should load models
        processor1, model1 = cache.get_owl()
        assert processor1 == mock_processor
        assert model1 == mock_model
        assert cache._owl_loaded

        # Second call should return the same instances without reloading
        processor2, model2 = cache.get_owl()
        assert processor2 is processor1
        assert model2 is model1

        # Should only call from_pretrained once
        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    def test_cleanup(self):
        """Test that cleanup properly resets the cache."""
        cache = CachedSceneModels(device="cpu")
        cache._owl_loaded = True
        cache._sam_loaded = True
        cache.owl_processor = MagicMock()
        cache.owl_model = MagicMock()

        cache.cleanup()

        assert cache.owl_processor is None
        assert cache.owl_model is None
        assert cache.sam_predictor is None
        assert not cache._owl_loaded
        assert not cache._sam_loaded


class TestDetectObjectsIntegration:
    """Test detect_objects function with mocked dependencies."""

    @patch('transformers.Owlv2Processor')
    @patch('transformers.Owlv2ForObjectDetection')
    @patch('torch.no_grad')
    @patch('torch.tensor')
    def test_detect_objects_with_cached_models(self, mock_tensor, mock_no_grad, mock_model_class, mock_processor_class):
        """Test detect_objects with CachedSceneModels."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock the detection results
        mock_tensor_obj = MagicMock()
        mock_tensor_obj.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_results = {
            'boxes': mock_tensor_obj,
            'scores': mock_tensor_obj,
            'labels': mock_tensor_obj
        }
        mock_processor.post_process_object_detection.return_value = [mock_results]
        mock_tensor.return_value = mock_tensor_obj
        mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)

        # Create test data
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        queries = ["cup"]
        cache = CachedSceneModels(device="cpu")

        # First call - should load models
        result1 = detect_objects(image, queries, cached_models=cache)
        assert isinstance(result1, list)
        assert cache._owl_loaded

        # Second call - should reuse models
        result2 = detect_objects(image, queries, cached_models=cache)
        assert isinstance(result2, list)

        # Should only load models once
        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    def test_detect_objects_without_cached_models(self):
        """Test detect_objects without cached models (should work but load models each time)."""
        # Create test data
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        queries = ["cup", "bowl"]

        with patch('transformers.Owlv2Processor') as mock_processor_class, \
             patch('transformers.Owlv2ForObjectDetection') as mock_model_class, \
             patch('torch.no_grad') as mock_no_grad, \
             patch('torch.tensor') as mock_tensor:

            # Setup basic mocks to prevent actual model loading
            mock_processor = MagicMock()
            mock_model = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model

            # Mock empty results
            mock_tensor_obj = MagicMock()
            mock_tensor_obj.cpu.return_value.numpy.return_value = np.array([])
            mock_processor.post_process_object_detection.return_value = [{
                'boxes': mock_tensor_obj,
                'scores': mock_tensor_obj,
                'labels': mock_tensor_obj
            }]
            mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)

            result = detect_objects(image, queries, device="cpu")
            assert isinstance(result, list)

    def test_detect_objects_empty_queries(self):
        """Test detect_objects with empty query list."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = detect_objects(image, [], device="cpu")
        assert result == []


class TestSegmentSceneIntegration:
    """Test segment_scene function with mocked dependencies."""

    def test_segment_scene_no_detections(self):
        """Test segment_scene with no detections."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = []

        result = segment_scene(image, detections, device="cpu")

        assert isinstance(result, SceneSegmentation)
        assert len(result.objects) == 0
        assert result.background_mask.shape == (100, 100)
        assert np.all(result.background_mask)  # Should be all background

    @patch('segment_anything.SamPredictor')
    @patch('segment_anything.sam_model_registry')
    @patch('smolvla_inspect.diagnostic.scene._ensure_sam_checkpoint')
    def test_segment_scene_with_cached_models(self, mock_checkpoint, mock_registry, mock_predictor_class):
        """Test segment_scene with CachedSceneModels."""
        # Setup mocks
        mock_sam = MagicMock()
        mock_predictor = MagicMock()
        mock_registry.__getitem__.return_value = lambda checkpoint: mock_sam
        mock_sam.to.return_value = mock_sam
        mock_predictor_class.return_value = mock_predictor
        mock_checkpoint.return_value = "/fake/path"

        # Mock SAM prediction results
        mock_predictor.predict.return_value = (
            np.array([np.ones((100, 100), dtype=bool)]),  # masks
            np.array([0.9]),  # scores
            None  # logits
        )

        # Create test data
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = [
            DetectedObject(label="cup", box=(10, 10, 50, 50), score=0.8, mask=None)
        ]
        cache = CachedSceneModels(device="cpu")

        result = segment_scene(image, detections, device="cpu", cached_models=cache)

        assert isinstance(result, SceneSegmentation)
        assert len(result.objects) == 1
        assert result.objects[0].mask is not None


@pytest.mark.integration
class TestEndToEndSceneProcessing:
    """Integration tests for the complete scene processing pipeline."""

    def test_complete_pipeline_with_caching(self):
        """Test the complete pipeline: parse -> detect -> segment with caching."""
        # Parse task objects
        task_objects = parse_task_objects("pick up the red block and place it in the blue cup")
        assert "robot gripper" in task_objects

        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test with mocked models to avoid actual model loading
        with patch('transformers.Owlv2Processor') as mock_processor_class, \
             patch('transformers.Owlv2ForObjectDetection') as mock_model_class, \
             patch('torch.no_grad') as mock_no_grad, \
             patch('torch.tensor') as mock_tensor:

            # Setup detection mocks
            mock_processor = MagicMock()
            mock_model = MagicMock()
            mock_processor_class.from_pretrained.return_value = mock_processor
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model

            # Mock empty detection results
            mock_tensor_obj = MagicMock()
            mock_tensor_obj.cpu.return_value.numpy.return_value = np.array([])
            mock_processor.post_process_object_detection.return_value = [{
                'boxes': mock_tensor_obj,
                'scores': mock_tensor_obj,
                'labels': mock_tensor_obj
            }]
            mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)

            # Create cached models and run detection
            cache = CachedSceneModels(device="cpu")
            detections = detect_objects(image, task_objects, cached_models=cache)

            # Run segmentation
            segmentation = segment_scene(image, detections, device="cpu", cached_models=cache)

            # Verify results
            assert isinstance(detections, list)
            assert isinstance(segmentation, SceneSegmentation)
            assert segmentation.image_shape == (100, 100)

            # Verify models were cached
            assert cache._owl_loaded

            # Cleanup
            cache.cleanup()
            assert not cache._owl_loaded


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_scene_models.py -v
    pytest.main([__file__, "-v"])