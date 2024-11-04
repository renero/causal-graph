import pytest
from unittest.mock import MagicMock
from causalgraph.estimators.rex import Rex
from causalgraph.common import DEFAULT_ITERATIVE_TRIALS

@pytest.fixture
def rex_instance():
    return Rex(name="test_rex")

def test_steps_from_iterations_with_iterative_predict_and_num_iterations(rex_instance):
    fit_steps = MagicMock()
    fit_steps.contains_method.return_value = 1
    fit_steps.contains_argument.return_value = True
    fit_steps.all_argument_values.return_value = [5, 10, 15]

    result = rex_instance._steps_from_iterations(fit_steps)
    assert result == 30  # Sum of [5, 10, 15]

def test_steps_from_iterations_with_iterative_predict_without_num_iterations(rex_instance):
    fit_steps = MagicMock()
    fit_steps.contains_method.return_value = 1
    fit_steps.contains_argument.return_value = False

    result = rex_instance._steps_from_iterations(fit_steps)
    assert result == DEFAULT_ITERATIVE_TRIALS

def test_steps_from_iterations_without_iterative_predict(rex_instance):
    fit_steps = MagicMock()
    fit_steps.contains_method.return_value = 0

    result = rex_instance._steps_from_iterations(fit_steps)
    assert result == 0