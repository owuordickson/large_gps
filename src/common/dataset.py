# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "2.2"
@email: "owuordickson@gmail.com"
@created: "05 February 2021"

Description
-------


"""


import csv
from dateutil.parser import parse
import time
import numpy as np


class Dataset:

    def __init__(self, file_path, min_sup=0):
        data = Dataset.read_csv(file_path)
        if len(data) <= 1:
            self.data = np.array([])
            print("csv file read error")
            raise Exception("Unable to read csv file or file has no data")
        else:
            print("Data fetched from csv file")
            self.thd_supp = min_sup
            self.data = np.array([])
            self.titles = self.get_titles(data)
            self.col_count = self.get_cols_count()
            self.row_count = self.get_size()
            self.time_cols = self.get_time_cols()
            self.attr_cols = self.get_attr_cols()
            self.no_bins = False
            self.seg_count = 1
            # self.attr_size = 0
            # self.step_name = ''
            self.invalid_bins = np.array([])
            self.valid_bins = np.array([])
            # self.d_matrix = np.array([])
            # self.p_matrix = np.array([])
            data = None
            # self.init_attributes()

    def get_titles(self, data):
        # data = self.raw_data
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            titles = self.convert_data_to_array(data)
            return titles
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                titles = self.convert_data_to_array(data)
                return titles
            else:
                titles = self.convert_data_to_array(data, has_title=True)
                return titles

    def get_size(self):
        size = self.data.shape[0]
        return size

    def get_cols_count(self):
        count = self.data.shape[1]
        return count

    def get_attr_cols(self):
        all_cols = np.arange(self.get_cols_count())
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        time_cols = list()
        # for k in range(10, len(self.data[0])):
        #    time_cols.append(k)
        # time_cols.append(0)
        n = len(self.data[0])
        for i in range(n):  # check every column for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = Dataset.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        if len(time_cols) > 0:
            return np.array(time_cols)
        else:
            return np.array([])

    def convert_data_to_array(self, data, has_title=False):
        # convert csv data into array
        titles = np.array([])
        if has_title:
            keys = np.arange(len(data[0]))
            values = np.array(data[0], dtype='S')
            titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
            data = np.delete(data, 0, 0)
        # convert csv data into array
        self.data = np.asarray(data)
        return titles

    def init_gp_attributes(self, seg_no):
        # (check) implement parallel multiprocessing
        # transpose csv array data
        attr_data = self.data.copy().T
        # construct and store 1-item_set valid bins
        self.construct_bins(attr_data, seg_no)
        attr_data = None
        # self.aco_code()

    def construct_bins(self, attr_data, seg_no):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        # n = self.attr_size
        valid_bins = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')
            arr_pos, arr_neg = self.bin_rank(col_data, seg_no)
            if arr_pos is None:
                invalid_bins.append(incr)
            else:
                valid_bins.append(np.array([incr.tolist(), arr_pos[0], arr_pos[1]], dtype=object))
            if arr_neg is None:
                invalid_bins.append(decr)
            else:
                valid_bins.append(np.array([decr.tolist(), arr_neg[0], arr_neg[1]], dtype=object))
        # print(valid_bins)
        self.valid_bins = np.array(valid_bins)
        self.invalid_bins = np.array(invalid_bins)
        if len(self.valid_bins) < 3:
            self.no_bins = True
        else:
            self.seg_count = self.valid_bins[0][1].size
            # valid_attr_count = self.valid_bins.size
            # self.d_matrix = np.zeros(valid_attr_count, self.seg_count)
            # self.d_matrix = np.stack(self.valid_bins[:, 1])
            # self.p_matrix = np.ones(self.d_matrix.shape, dtype=float)
            # print(self.d_matrix)
            # print(self.d_matrix.T)
            # print("-------\n\n")

    def bin_rank(self, arr, seg_no):
        n = self.row_count
        step = int(self.row_count / seg_no)
        lst_pos = []
        lst_neg = []
        lst_pos_sum = []
        lst_neg_sum = []
        with np.errstate(invalid='ignore'):
            for i in range(0, n, step):
                if i == 0:
                    bin_neg = arr < arr[:step, np.newaxis]
                    bin_pos = arr > arr[:step, np.newaxis]
                else:
                    if (i+step) < n:
                        bin_neg = arr < arr[i:(i+step), np.newaxis]
                        bin_pos = arr > arr[i:(i+step), np.newaxis]
                    else:
                        bin_neg = arr < arr[i:, np.newaxis]
                        bin_pos = arr > arr[i:, np.newaxis]
                lst_neg.append(bin_neg)
                lst_pos.append(bin_pos)
                lst_neg_sum.append(np.sum(bin_neg))
                lst_pos_sum.append(np.sum(bin_pos))
            sup_neg = float(np.sum(lst_neg_sum)) / float(n * (n - 1.0) / 2.0)
            sup_pos = float(np.sum(lst_pos_sum)) / float(n * (n - 1.0) / 2.0)
            if sup_neg < self.thd_supp:
                lst_neg = None
            else:
                lst_neg = [np.array(lst_neg_sum, dtype=int), np.array(lst_neg, dtype=object)]
            if sup_pos < self.thd_supp:
                lst_pos = None
            else:
                lst_pos = [np.array(lst_pos_sum, dtype=int), np.array(lst_pos, dtype=object)]
            return lst_pos, lst_neg

    def aco_code_n(self):
        d = self.d_matrix
        p = self.p_matrix
        e = .5  # evaporation factor

        with np.errstate(divide='ignore'):
            # calculating the visibility of the next city visibility(i,j)=1/d(i,j)
            visibility = 1/d
            visibility[visibility == np.inf] = 0
            print(visibility)

    def aco_code(self):
        d = self.d_matrix
        iteration = 100
        n_ants = 8
        n_city = 3
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

    @staticmethod
    def read_csv(file):
        # 1. retrieve data-set from file
        with open(file, 'r') as f:
            dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
            f.seek(0)
            reader = csv.reader(f, dialect)
            temp = list(reader)
            f.close()
        return temp

    @staticmethod
    def test_time(date_str):
        # add all the possible formats
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('no valid date-time format found')

    @staticmethod
    def get_timestamp(time_data):
        try:
            ok, stamp = Dataset.test_time(time_data)
            if ok:
                return stamp
            else:
                return False
        except ValueError:
            return False
