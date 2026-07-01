"""
New tests added by Evyatar Reif
"""

from __future__ import annotations
import pytest

def test_nested_loop_accuracy(predictor_module, traces_dir):
    """Test that a local-local predictor can learn a nested loop pattern."""
    
    # 1. Import the evaluator
    from src.lab_utils import evaluate_predictor
    
    # 2. Point to the new trace file you just created
    trace_file = str(traces_dir / "nested_loop.trace")

    # 3. Configure the predictor you want to test
    # A local predictor is great for loops. We need at least history_size=4 
    # to remember the "T T T N" pattern of the inner loop.
    cfg = predictor_module.PredictorConfig(
        history_size=4, 
        history_table_size=16,
        history_mode="local", 
        table_mode="local", 
        share=False
    )

    # 4. Run the evaluation
    results = evaluate_predictor(cfg, trace_file, predictor_module)

    # 5. Assert the expected behavior
    # Results dictionary usually contains "total", "correct", "mispredicted", and "accuracy".
    print(f"Nested loop accuracy: {results['accuracy']}")
    
    # We expect high accuracy on predictable loops!
    assert results["accuracy"] > 0.80, "Predictor failed to learn the nested loop!"