"""
Methods in this module perform HSIC independence test, and compute the HSIC
value and statistic.

After checking that most implementations fail to provide consistent results,
I decided to take the implementation from the DoWhy package.
"""
from dataclasses import dataclass
from dowhy.gcm.independence_test import independence_test

@dataclass
class HSIC_Values:
    hsic: float = 0.
    p_value: float = 0.
    stat: float = 0.
    independence: bool = False

    def __init__(self, hsic, p_value, stat, independence):
        self.hsic = hsic
        self.p_value = p_value
        self.stat = stat
        self.independence = independence

    def __str__(self):
        s = f"HSIC...: {self.hsic:.6g}\n"
        s += f"p_value: {self.p_value:.6g}\n"
        s += f"stat...: {self.stat:.6g}\n"
        s += f"indep..: {self.independence}"
        return s


class HSIC:
    """
    Provides a class sklearn-type interface to the Hsic methods in this DoWhy.
    """
    def __init__(self):
        self.p_val: float = 0.0
        self.h_val: float = 0.0
        self.h_stat: float = 0.0
        self.h_indep: bool = False

    def fit(self, X, Y):
        self.p_val = independence_test(X, Y, conditioned_on=None, method="kernel")
        self.h_indep = self.p_val > 0.05
        return HSIC_Values(self.h_val, self.p_value, self.h_stat, self.h_indep)

    @property
    def p_value(self):
        return self.p_val

    @property
    def hsic(self):
        return self.h_val

    @property
    def stat(self):
        return self.h_stat

    @property
    def independence(self):
        return self.h_indep
