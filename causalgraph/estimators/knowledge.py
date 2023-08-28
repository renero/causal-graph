import math
import numpy as np
import pandas as pd


class Knowledge:
    """
    This class collects everything we know about each edge in the proposed graph
    in terms of the following properties:

    - origin: the origin node
    - target: the target node
    - ref_edge: whether the edge is in the reference graph
    - correlation: the correlation between the individual SHAP values and the origin node
    - KS_pval: the p-value of the Kolmogorov-Smirnov test between the origin and the target
    - shap_edge: whether the edge is in the graph constructed after evaluating mean 
        SHAP values.
    - shap_skedastic_pval: the p-value of the skedastic test for the SHAP values
    - parent_skedastic_pval: the p-value of the skedastic test for the parent values
    - mean_shap: the mean of the SHAP values between the origin and the target
    - slope_shap: the slope of the linear regression for target vs. SHAP values
    - slope_target: the slope of the linear regression for the target vs. origin values

    """

    def __init__(self, rex: Rex, ref_graph: nx.DiGraph):
        """
        Arguments:
        ----------
            shaps (ShapEstimator): The shap estimator.
            ref_graph (nx.DiGraph): The reference graph, or ground truth.    
        """
        self.K = 180.0 / math.pi
        self.shaps = rex.shaps
        self.pi = rex.pi
        self.hierarchies = rex.hierarchies
        self.feature_names = rex.feature_names
        self.scoring = rex.models.scoring
        self.ref_graph = ref_graph

    def _compute_regression_outliers(self):
        """
        Determine what features are outliers in the regression, according to their
        score with a test set after fitting the model. The criteria to determine
        whether a feature is an outlier is whether its score is greater than 2.5
        times the interquartile range (IQR).
        """
        iqr = np.quantile(self.scoring, 0.75) - np.quantile(self.scoring, 0.25)
        outliers_indices = np.where(
            np.abs(self.scoring - np.median(self.scoring)) > 1.5*iqr)
        self.regression_outliers = [self.feature_names[i]
                                    for i in outliers_indices[0]]

    def data(self):
        """Returns a dataframe with the knowledge about each edge in the graph"""
        rows = []
        self._compute_regression_outliers()
        for origin in self.feature_names:
            for target in self.feature_names:
                all_origins = [
                    o for o in self.feature_names if o != target]
                all_targets = [
                    t for t in self.feature_names if t != origin]
                if origin != target:
                    origin_pos = all_origins.index(origin)
                    target_pos = all_targets.index(target)

                    sd = self.shaps.shap_discrepancies[origin][target]
                    pi = self.pi.pi[origin]['importances_mean'][target_pos]

                    b0_s, b1_s = sd.shap_model.params[0], sd.shap_model.params[1]
                    b0_y, b1_y = sd.parent_model.params[0], sd.parent_model.params[1]
                    shap_slope = math.atan(b1_s)*self.K
                    parent_slope = math.atan(b1_y)*self.K
                    rows.append({
                        'origin': origin,
                        'target': target,
                        'ref_edge': int((origin, target) in self.ref_graph.edges()),
                        'correlation': self.hierarchies.correlations[origin][target],
                        'shap_correlation': sd.shap_correlation,
                        'KS_pval': sd.ks_pvalue,
                        'shap_edge': int(origin in self.shaps.parents[target]),
                        'shap_skedastic_pval': sd.shap_p_value,
                        'parent_skedastic_pval': sd.parent_p_value,
                        'mean_shap': self.shaps.shap_mean_values[target][origin_pos],
                        'mean_pi': pi,
                        'slope_shap': shap_slope,
                        'slope_target': parent_slope,
                        'potential_root': int(origin in self.regression_outliers)
                    })
        self.results = pd.DataFrame.from_dict(rows)
        return self.results
