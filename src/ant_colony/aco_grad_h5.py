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
from common.gp import GI, GP
from common.dataset import Dataset


class GradACO:

    def __init__(self, f_path, min_supp, segs):
        self.d_set = Dataset(f_path, min_supp)
        self.d_set.init_gp_attributes(segs)
        self.attr_index = self.d_set.attr_cols
        self.e_factor = 0.5  # evaporation factor
        self.p_matrix = np.ones((self.d_set.col_count, 3), dtype=float)

    def deposit_pheromone(self, pattern):
        lst_attr = []
        for obj in pattern.gradual_items:
            # print(obj.attribute_col)
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] += 1
            elif symbol == '-':
                self.p_matrix[i][1] += 1
        for index in self.attr_index:
            if int(index) not in lst_attr:
                i = int(index)
                self.p_matrix[i][2] += 1

    def evaporate_pheromone(self, pattern):
        lst_attr = []
        for obj in pattern.gradual_items:
            # print(obj.attribute_col)
            attr = obj.attribute_col
            symbol = obj.symbol
            lst_attr.append(attr)
            i = attr
            if symbol == '+':
                self.p_matrix[i][0] = (1 - self.e_factor) * self.p_matrix[i][0]
            elif symbol == '-':
                self.p_matrix[i][1] = (1 - self.e_factor) * self.p_matrix[i][1]

    def run_ant_colony(self):
        min_supp = self.d_set.thd_supp
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        if self.d_set.no_bins:
            return []
        while repeated < 1:
            rand_gp = self.generate_random_gp()
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
                            self.deposit_pheromone(gen_gp)
                            winner_gps.append(gen_gp)
                        else:
                            loser_gps.append(gen_gp)
                            # update pheromone as irrelevant with loss_sols
                            self.evaporate_pheromone(gen_gp)
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
        return winner_gps

    def generate_random_gp(self):
        p = self.p_matrix
        n = len(self.attr_index)
        pattern = GP()
        attrs = np.random.permutation(n)
        for i in attrs:
            max_extreme = n * 100
            x = float(rand.randint(1, max_extreme) / max_extreme)
            pos = float(p[i][0] / (p[i][0] + p[i][1] + p[i][2]))
            neg = float((p[i][0] + p[i][1]) / (p[i][0] + p[i][1] + p[i][2]))
            if x < pos:
                temp = GI(self.attr_index[i], '+')
            elif (x >= pos) and (x < neg):
                temp = GI(self.attr_index[i], '-')
            else:
                # temp = GI(self.attr_index[i], 'x')
                continue
            pattern.add_gradual_item(temp)
        return pattern

    def validate_gp_old(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        bin_data = np.array([])

        for gi in pattern.gradual_items:
            if self.d_set.invalid_bins.size > 0 and np.any(np.isin(self.d_set.invalid_bins, gi.gradual_item)):
                continue
            else:
                print(self.d_set.valid_bins[:, 0])
                arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], gi.gradual_item))
                # print(arg)
                arg = [[2]]
                if len(arg) > 0:
                    i = arg[0][0]
                    bin_obj = self.d_set.valid_bins[i]
                    if bin_data.size <= 0:
                        bin_data = np.array([bin_obj[1], bin_obj[1]])
                        gen_pattern.add_gradual_item(gi)
                    else:
                        bin_data[1] = bin_obj[1]
                        temp_bin, supp = self.bin_and(bin_data, self.d_set.row_count)
                        if supp >= min_supp:
                            bin_data[0] = temp_bin
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

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

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        n = self.d_set.row_count
        min_supp = self.d_set.thd_supp
        gen_pattern = GP()
        arg = np.argwhere(np.isin(self.d_set.valid_bins[:, 0], pattern.get_np_pattern()))
        if len(arg) >= 2:
            bin_data = self.d_set.valid_bins[arg.flatten()]
            seg_sum = np.sum(bin_data[:, 1], axis=0)
            seg_order = np.argsort(-seg_sum)
            bin_arr = bin_data[:, 2]

            print(seg_order)
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
                        supp = float(bin_sum + temp_sum) / float(n * (n - 1.0) / 2.0)  # TO BE REMOVED
                        # print(temp_bin)
                        # print("\n")
                        print(str(i) + " -- sum: " + str(temp_sum) + ' | total: ' + str(bin_sum))
                        print("Support: " + str(supp))

                        if not gen_pattern.contains(gi_1):
                            gen_pattern.add_gradual_item(gi_1)
                        gi_2 = GI(bin_data[j][0][0], bin_data[j][0][1].decode())
                        if not gen_pattern.contains(gi_2):
                            print(gi_2.to_string())
                            print(gen_pattern.to_string())
                            gen_pattern.add_gradual_item(gi_2)
                        gen_pattern.set_support(supp)
                bin_sum += temp_sum
                supp = float(bin_sum) / float(n * (n - 1.0) / 2.0)
                if supp >= min_supp:
                    # self.
                    print("stopped at: " + str(i))
            # print(seg_sum)
            # print(seg_order)
            # print(bin_arr)
            # print(self.d_set.valid_bins.shape)
            # print(bin_data.shape)
            # print(bins[:, 0])
            # print(pattern.get_np_pattern())
            # print(bin_obj)
            # print("----\n\n")
            # self.bin_and(arg)
            # print(self.d_set.valid_bins[:, 0])
            # print(arg)
            # print(len(arg))
        print(gen_pattern.to_string())
        print("----\n\n")
        if len(gen_pattern.gradual_items) <= 1:
            return pattern
        else:
            return gen_pattern

    def bin_and(self, bins, n):
        # bin_ = np.zeros((n, n), dtype=bool)
        bin_obj1 = bins[0]
        bin_obj2 = bins[1]
        # seg_order = GradACO.obtain_seg_order(bin_obj1[0], bin_obj2[0])  TO BE REMOVED
        seg_order = np.argsort(-(bin_obj1[0] + bin_obj2[0]))
        for i in seg_order:
            temp_bin = bin_obj1[1][i] * bin_obj2[1][i]
            supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
            if supp >= self.d_set.thd_supp:
                print(supp)
                print(i)
                print(temp_bin)
                break
        # temp_bin = bins[0] * bins[1]
        # supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
        return bins[0], 0  # TO BE REMOVED

    @staticmethod  # TO BE REMOVED
    def obtain_seg_order(seg1, seg2):
        # print(seg1)
        # print(seg2)
        order = np.argsort(-(seg1 + seg2))
        # print((seg1 + seg2))
        # print(order)
        # print("----- \n\n")
        return order
