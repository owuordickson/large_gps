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
# import pandas as pd
from numpy import random as rand
from common.gp import GI, GP
from common.dataset import Dataset


class GradACO:

    def __init__(self, f_path, min_supp, segs):
        self.d_set = Dataset(f_path, segs, min_supp)
        self.attr_index = self.d_set.attr_cols
        self.e_factor = 0.5  # evaporation factor
        self.iteration_count = 0
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        p_matrix = self.d_set.read_h5_dataset(grp)
        if np.sum(p_matrix) > 0:
            self.p_matrix = p_matrix
        else:
            self.p_matrix = np.ones((self.d_set.col_count, 3), dtype=float)
        # self.p_matrix = np.ones((self.d_set.col_count, 3), dtype=float)
        # self.aco_code()

    def update_pheromone(self, pattern):
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

    def run_ant_colony_old(self):
        min_supp = self.d_set.thd_supp
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        it_count = 0
        if self.d_set.no_bins:
            return []
        # while repeated < 1:
        while it_count <= 10:
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
                            self.update_pheromone(gen_gp)
                            winner_gps.append(gen_gp)
                        else:
                            loser_gps.append(gen_gp)
                            # update pheromone as irrelevant with loss_sols
                            self.evaporate_pheromone(gen_gp)
                    if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                        loser_gps.append(rand_gp)
                else:
                    repeated += 1
            it_count += 1
        self.iteration_count = it_count
        grp = 'dataset/' + self.d_set.step_name + '/p_matrix'
        self.d_set.add_h5_dataset(grp, self.p_matrix)
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

    def validate_gp(self, pattern):
        # pattern = [('2', '+'), ('4', '+')]
        n = self.d_set.row_count
        min_supp = self.d_set.thd_supp
        seg_count = 0
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
        # 1. Fetch valid bins group
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
        return d

    def run_ant_colony(self):
        min_supp = self.d_set.thd_supp
        winner_gps = list()  # subsets
        loser_gps = list()  # supersets
        repeated = 0
        it_count = 0

        if self.d_set.no_bins:
            return []

        # 1. Retrieve/Generate distance matrix (d)
        grp_name = 'dataset/' + self.d_set.step_name + '/d_matrix'
        d = self.d_set.read_h5_dataset(grp_name)
        if d.size <= 0:
            d = self.generate_d()

        # 2. Remove d[i][j] < frequency-count of min_supp
        a = self.d_set.attr_size
        fr_count = ((min_supp * a * (a - 1)) / 2)
        d[d < fr_count] = 0
        print(d)

        # 3. Calculating the visibility of the next city visibility(i,j)=1/d(i,j)
        with np.errstate(divide='ignore'):
            visibility = 1/d
            visibility[visibility == np.inf] = 0

        # 4. Initialize pheromones (p_matrix)
        grp_name = 'dataset/' + self.d_set.step_name + '/p_matrix'
        pheromones = self.d_set.read_h5_dataset(grp_name)
        if pheromones.size <= 0:
            pheromones = np.ones(d.shape, dtype=float)
        print(pheromones)

        # 5. Iterations for ACO
        # while repeated < 1:
        while it_count <= 10:
            rand_gp = self.generate_gp(visibility, pheromones)
            it_count += 1

        # TO BE MOVED TO UPDATE-METHOD
        # Save pheromone matrix (p_matrix)
        self.iteration_count = it_count
        grp_name = 'dataset/' + self.d_set.step_name + '/p_matrix'
        self.d_set.add_h5_dataset(grp_name, pheromones)
        return winner_gps

    def generate_gp(self, v_matrix, p_matrix):
        pattern = GP()

        # 1. Fetch attributes corresponding to v_matrix and p_matrix
        grp_name = 'dataset/' + self.d_set.step_name + '/valid_bins/'
        h5f = h5py.File(self.d_set.h5_file, 'r')
        attr_keys = list(h5f[grp_name].keys())
        h5f.close()

        # 2.
        m, n = p_matrix.shape
        for i in range(m):
            for j in range(n):
                continue

        return pattern

    def aco_code(self):
        d = self.generate_d()
        iteration = 100
        n_ants = 8
        n_city = 8
        m = n_ants
        n = n_city
        e = .5  # evaporation factor
        alpha = 1  # pheromone factor
        beta = 2  # visibility factor

        # calculating the visibility of the next city visibility(i,j)=1/d(i,j)
        visibility = 1/d
        visibility[visibility == np.inf] = 0

        # intializing pheromne present at the paths to the cities

        pheromne = .1 * np.ones((m, n))

        # intializing the rute of the ants with size rute(n_ants,n_citys+1)
        # note adding 1 because we want to come back to the source city

        rute = np.ones((m, n + 1))

        for ite in range(iteration):

            rute[:, 0] = 1  # initial starting and ending positon of every ants '1' i.e city '1'

            for i in range(m):

                temp_visibility = np.array(visibility)  # creating a copy of visibility

                for j in range(n - 1):
                    # print(rute)

                    cur_loc = int(rute[i, j] - 1)  # current city of the ant

                    temp_visibility[:, cur_loc] = 0  # making visibility of the current city as zero

                    p_feature = np.power(pheromne[cur_loc, :], beta)  # calculating pheromne feature
                    v_feature = np.power(temp_visibility[cur_loc, :], alpha)  # calculating visibility feature

                    p_feature = p_feature[:, np.newaxis]  # adding axis to make a size[5,1]
                    v_feature = v_feature[:, np.newaxis]  # adding axis to make a size[5,1]

                    combine_feature = np.multiply(p_feature, v_feature)  # calculating the combine feature

                    total = np.sum(combine_feature)  # sum of all the feature

                    probs = combine_feature / total  # finding probability of element probs(i) = comine_feature(i)/total

                    cum_prob = np.cumsum(probs)  # calculating cummulative sum
                    # print(cum_prob)
                    r = np.random.random_sample()  # randon no in [0,1)
                    # print(r)
                    city = np.nonzero(cum_prob > r)[0][
                               0] + 1  # finding the next city having probability higher then random(r)
                    # print(city)

                    rute[i, j + 1] = city  # adding city to route

                left = list(set([i for i in range(1, n + 1)]) - set(rute[i, :-2]))[
                    0]  # finding the last untraversed city to route

                rute[i, -2] = left  # adding untraversed city to route

            rute_opt = np.array(rute)  # intializing optimal route

            dist_cost = np.zeros((m, 1))  # intializing total_distance_of_tour with zero

            for i in range(m):

                s = 0
                for j in range(n - 1):
                    s = s + d[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1]  # calcualting total tour distance

                dist_cost[i] = s  # storing distance of tour for 'i'th ant at location 'i'

            dist_min_loc = np.argmin(dist_cost)  # finding location of minimum of dist_cost
            dist_min_cost = dist_cost[dist_min_loc]  # finging min of dist_cost

            best_route = rute[dist_min_loc, :]  # intializing current traversed as best route
            pheromne = (1 - e) * pheromne  # evaporation of pheromne with (1-e)

            for i in range(m):
                for j in range(n - 1):
                    dt = 1 / dist_cost[i]
                    pheromne[int(rute_opt[i, j]) - 1, int(rute_opt[i, j + 1]) - 1] = pheromne[
                                                                                         int(rute_opt[i, j]) - 1, int(
                                                                                             rute_opt[
                                                                                                 i, j + 1]) - 1] + dt
                    # updating the pheromne with delta_distance
                    # delta_distance will be more with min_dist i.e adding more weight to that route  peromne

        print('route of all the ants at the end :')
        print(rute_opt)
        print()
        print('best path :', best_route)
        print('cost of the best path', int(dist_min_cost[0]) + d[int(best_route[-2]) - 1, 0])