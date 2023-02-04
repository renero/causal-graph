import pytest

from sklearn.utils.estimator_checks import check_estimator

from causalgraph import TemplateEstimator
from causalgraph import TemplateClassifier
from causalgraph import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
