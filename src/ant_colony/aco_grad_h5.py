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
import h5py
import numpy as np
from itertools import combinations
# import pandas as pd
from common.gp import GI, GP
from common.dataset import Dataset


class GradACO:

    def __init__(self, f_path, min_supp, segs):
        self.d_set = Dataset(f_path, segs, min_supp)
        self.attr_index = self.d_set.attr_cols
        self.e_factor = 0.5  # evaporation factor
        self.iteration_count = 0
        self.d, self.attr_keys = self.generate_d()  # distance matrix (d) & attributes corresponding to d

    def bin_and(self, bins):
        n = self.d_set.row_count
        bin_1 = bins[0]
        bin_2 = bins[1]
        temp_bin = []
        bin_sum = 0
        for i in range(len(bin_1)):
            prod = bin_1[i] * bin_2[i]
            temp_bin.append(prod)
            bin_sum += np.sum(prod)
            # print(str(bin_1[i]) + ' x ' + str(bin_2[i]) + '\n')
            # print(temp_bin)
            # print("***")
        temp_bin = np.array(temp_bin, dtype=object)
        # print(temp_bin)
        # print(np.sum(temp_bin))
        supp = float(bin_sum) / float(n * (n - 1.0) / 2.0)
        return temp_bin, supp

    @staticmethod
    def check_anti_monotony(lst_p, pattern, subset=True):
        result = False
        if subset:
            for pat in lst_p:
                result1 = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
                result2 = set(pattern.inv_pattern()).issubset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in lst_p:
                result1 = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
                result2 = set(pattern.inv_pattern()).issuperset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        return result

    @staticmethod
    def is_duplicate(pattern, lst_winners, lst_losers):
        for pat in lst_losers:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
        for pat in lst_winners:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
        return False

    # ADDED TEST CODES
    def validate_gp_wait(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        n = self.d_set.row_count
        min_supp = self.d_set.thd_supp
        seg_count = 0
        gen_pattern = GP()
        arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], pattern.get_np_pattern()))
        if len(arg) >= 2:
            bin_data = self.d_set.valid_bins[arg.flatten()]
            seg_sum = np.sum(bin_data[:, 1], axis=0)
            seg_order = np.argsort(-seg_sum)
            bin_arr = bin_data[:, 2]

            # print(seg_order)
            # print(bin_data)
            # print("\n")
            # print(bin_arr)
            # temp_bin = bin_arr[0][seg_order[0]]
            bin_sum = 0
            for i in seg_order:
                curr_bin = None
                gi_1 = None
                temp_sum = 0
                for j in range(len(bin_arr)):
                    if curr_bin is None:
                        curr_bin = bin_arr[j][i]
                        gi_1 = GI(bin_data[j][0][0], bin_data[j][0][1].decode())
                    else:
                        temp_bin = np.multiply(curr_bin, bin_arr[j][i])
                        temp_sum = np.sum(temp_bin)
                        curr_bin = np.copy(temp_bin)

                        # bin_sum += temp_sum
                        # supp = float(bin_sum + temp_sum) / float(n * (n - 1.0) / 2.0)  # TO BE REMOVED
                        # print(temp_bin)
                        # print("\n")
                        # print(str(i) + " -- sum: " + str(temp_sum) + ' | total: ' + str(bin_sum))
                        # print("Support: " + str(supp))

                        if not gen_pattern.contains(gi_1):
                            gen_pattern.add_gradual_item(gi_1)
                        gi_2 = GI(bin_data[j][0][0], bin_data[j][0][1].decode())
                        if not gen_pattern.contains(gi_2):
                            # print(gi_2.to_string())
                            # print(gen_pattern.to_string())
                            gen_pattern.add_gradual_item(gi_2)
                        supp = float(bin_sum + temp_sum) / float(n * (n - 1.0) / 2.0)
                        gen_pattern.set_support(supp)
                bin_sum += temp_sum
                seg_count += 1
                supp = float(bin_sum) / float(n * (n - 1.0) / 2.0)
                if supp >= min_supp:
                    # print("stopped at: " + str(seg_count))
                    break
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

    def generate_d(self):
        # 1a. Retrieve/Generate distance matrix (d)
        grp_name = 'dataset/' + self.d_set.step_name + '/d_matrix'
        d = self.d_set.read_h5_dataset(grp_name)
        if d.size > 0:
            # 1b. Fetch valid bins group
            grp_name = 'dataset/' + self.d_set.step_name + '/valid_bins/'
            h5f = h5py.File(self.d_set.h5_file, 'r')
            attr_keys = list(h5f[grp_name].keys())
            h5f.close()
            return d, attr_keys

        # 1b. Fetch valid bins group
        grp_name = 'dataset/' + self.d_set.step_name + '/valid_bins/'
        h5f = h5py.File(self.d_set.h5_file, 'r')
        grp = h5f[grp_name]
        attr_keys = list(grp.keys())

        # 2. Initialize an empty d-matrix
        n = len(grp)
        d = np.zeros((n, n), dtype=float)  # cumulative sum of all segments
        for k in range(self.d_set.seg_count):
            # 2. For each segment do a binary AND
            for i in range(n):
                for j in range(n):
                    bin_1 = grp[attr_keys[i]]
                    bin_2 = grp[attr_keys[j]]
                    if GI.parse_gi(attr_keys[i]).attribute_col == GI.parse_gi(attr_keys[j]).attribute_col:
                        # Ignore similar attributes (+ or/and -)
                        continue
                    else:
                        # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                        d[i][j] += np.sum(np.multiply(bin_1['bins'][str(k)][:], bin_2['bins'][str(k)][:], ))

        # 3. Save d_matrix in HDF5 file
        h5f.close()
        grp_name = 'dataset/' + self.d_set.step_name + '/d_matrix'
        self.d_set.add_h5_dataset(grp_name, d)
        return d, attr_keys

    def run_ant_colony(self):
        min_supp = self.d_set.thd_supp
        d = self.d
        a = self.d_set.attr_size
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        it_count = 0

        if self.d_set.no_bins:
            return []

        # 1. Remove d[i][j] < frequency-count of min_supp
        fr_count = ((min_supp * a * (a - 1)) / 2)
        d[d < fr_count] = 0
        print(d)

        # 2. Calculating the visibility of the next city
        # visibility(i,j)=1/d(i,j)
        # In the case GP mining visibility = d
        # with np.errstate(divide='ignore'):
        #    visibility = 1/d
        #    visibility[visibility == np.inf] = 0

        # 3. Initialize pheromones (p_matrix)
        pheromones = np.ones(d.shape, dtype=float)
        print(pheromones)
        print("***\n")

        # 4. Iterations for ACO
        # while repeated < 1:
        while it_count <= 5:
            rand_gp, pheromones = self.generate_aco_gp(pheromones)
            if len(rand_gp.gradual_items) > 1:
                # print(rand_gp.get_pattern())
                exits = GradACO.is_duplicate(rand_gp, winner_gps, loser_gps)
                if not exits:
                    repeated = 0
                    # check for anti-monotony
                    is_super = GradACO.check_anti_monotony(loser_gps, rand_gp, subset=False)
                    is_sub = GradACO.check_anti_monotony(winner_gps, rand_gp, subset=True)
                    if is_super or is_sub:
                        continue
                    gen_gp = self.validate_gp(rand_gp)
                    is_present = GradACO.is_duplicate(gen_gp, winner_gps, loser_gps)
                    is_sub = GradACO.check_anti_monotony(winner_gps, gen_gp, subset=True)
                    if is_present or is_sub:
                        repeated += 1
                    else:
                        if gen_gp.support >= min_supp:
                            pheromones = self.update_pheromones(gen_gp, pheromones)
                            winner_gps.append(gen_gp)
                        else:
                            loser_gps.append(gen_gp)
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
            it_count += 1

        self.iteration_count = it_count
        # print(pheromones)
        # print("***\n")
        return winner_gps

    def generate_aco_gp(self, p_matrix):
        attr_keys = self.attr_keys
        v_matrix = self.d
        pattern = GP()

        # 1. Generate gradual items with highest pheromone and visibility
        m = p_matrix.shape[0]
        for i in range(m):
            combine_feature = np.multiply(v_matrix[i], p_matrix[i])
            total = np.sum(combine_feature)
            with np.errstate(divide='ignore', invalid='ignore'):
                probability = combine_feature / total
            cum_prob = np.cumsum(probability)
            r = np.random.random_sample()
            try:
                j = np.nonzero(cum_prob > r)[0][0]
                gi = GI.parse_gi(attr_keys[j])
                if not pattern.contains_attr(gi):
                    pattern.add_gradual_item(gi)
            except IndexError:
                continue
        print("Generated pattern: " + str(pattern.to_string()))

        # 2. Evaporate pheromones by factor e
        p_matrix = (1 - self.e_factor) * p_matrix
        return pattern, p_matrix

    def update_pheromones(self, pattern, p_matrix):
        v_matrix = self.d
        idx = [self.attr_keys.index(x.as_string()) for x in pattern.gradual_items]
        combs = list(combinations(idx, 2))
        print(idx)
        print(combs)
        for i, j in combs:
            if v_matrix[i][j] > 0:
                p_matrix[i][j] += 1
            if v_matrix[j][i] > 0:
                p_matrix[j][i] += 1
        print(p_matrix)
        return p_matrix

    def validate_gp(self, pattern):
        # return GP()
        # pattern = [('2', '+'), ('4', '+')]
        # n = self.d_set.row_count
        min_supp = self.d_set.thd_supp
        # seg_count = 0
        gen_pattern = GP()
        arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], pattern.get_np_pattern()))
        if len(arg) >= 2:
            bin_data = self.d_set.valid_bins[arg.flatten()]
            segments = np.stack(bin_data[:, 1])
            large_col_sum = segments.sum(axis=0)
            large_rows = np.argsort(-segments[:, large_col_sum.argmax()])
            bin_arr = None
            for idx in large_rows:
                if bin_arr is None:
                    bin_arr = np.array([bin_data[idx][2], None], dtype=object)
                    gi = GI(bin_data[idx][0][0], bin_data[idx][0][1].decode())
                    gen_pattern.add_gradual_item(gi)
                    continue
                else:
                    bin_arr[1] = bin_data[idx][2]
                    temp_bin, supp = self.bin_and(bin_arr)
                    if supp >= min_supp:
                        bin_arr[0] = temp_bin
                        gi = GI(bin_data[idx][0][0], bin_data[idx][0][1].decode())
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)

        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

