#
# This file contains the methods used to estimate the effect of a treatment in
# an outcome, given a graph.
#
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate


def estimate(digraph, treatment, outcome, data, verbose=False) -> CausalEstimate:
    # Definir el modelo causal
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=digraph
    )

    # Identificar el efecto causal
    identified_estimand = model.identify_effect()
    print(identified_estimand) if verbose else None

    # Estimar el efecto causal
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    print(estimate) if verbose else None

    # Pruebas de refutación
    refute = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause"
    )
    print(refute) if verbose else None
    return estimate
