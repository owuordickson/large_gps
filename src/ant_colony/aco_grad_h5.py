# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent,"
@license: "MIT"
@version: "3.2"
@email: "owuordickson@gmail.com"
@created: "05 February 2021"

Breath-First Search for gradual patterns (ACO-GRAANK)

"""


import numpy as np
from numpy import random as rand
from ..common.gp import GI, GP
from ..common.dataset import Dataset


class GradACO:

    def __init__(self, f_path, min_supp, eq):
        self.d_set = Dataset(f_path, min_supp, eq)
        # self.d_set.init_gp_attributes()
        self.attr_index = self.d_set.attr_cols
        # self.e_factor = 0.5  # evaporation factor
        # self.p_matrix = np.ones((self.d_set.column_size, 3), dtype=float)