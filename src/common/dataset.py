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
from numpy.core._multiarray_umath import ndarray


class Dataset:

    def __init__(self, file_path, min_sup=0):
        data = Dataset.read_csv(file_path)
        if len(data) <= 1:
            self.data = np.array([])
            data = None
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
        # attr_cols = np.delete(all_cols, self.time_cols)
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
        titles: ndarray = np.array([])
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
        # self.attr_size = len(attr_data[self.attr_cols[0]])
        # construct and store 1-item_set valid bins
        self.construct_bins_new(attr_data, seg_no)
        attr_data = None

    def construct_bins(self, attr_data):
        ##print(self.data)
        #print("\n")
        #print(attr_data)
        #print("\n\n\n\n")
        # from itertools import product
        # ind = [[x, y] for x, y in product(range(n), range(n))]
        # arr_n = np.arange(n)
        # ind =
        # print(arr_n)
        # ind = [[i, j] for i in range(n) for j in range(i+1, n)]
        # print(ind)
        # print(len(ind))

        # generate tuple indices
        n = self.row_count #* 500
        valid_bins = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')
            #arr_pos = [col_data[i] > col_data[j] for i in range(n) for j in range(i + 1, n)]
            #arr_neg = [col_data[i] < col_data[j] for i in range(n) for j in range(i + 1, n)]
            arr_pos = [] #np.zeros((n), dtype=bool)
            arr_neg = [] #np.zeros((n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if col_data[i] > col_data[j]:
                        arr_pos.append(True)
                        arr_neg.append(False)
                    else:
                        if col_data[i] < col_data[j]:
                            arr_neg.append(True)
                            arr_pos.append(False)
                        else:
                            if self.equal:
                                arr_pos.append(True)
                                arr_neg.append(True)
            arr_pos = np.array(arr_pos)
            arr_neg = np.array(arr_neg)
            supp = float(np.sum(arr_pos)) / float(n * (n - 1.0) / 2.0)
            if supp < self.thd_supp:
                invalid_bins.append(incr)
            else:
                valid_bins.append(np.array([incr.tolist(), arr_pos], dtype=object))

            supp = float(np.sum(arr_neg)) / float(n * (n - 1.0) / 2.0)
            if supp < self.thd_supp:
                invalid_bins.append(decr)
            else:
                valid_bins.append(np.array([decr.tolist(), arr_neg], dtype=object))
        print(np.array(valid_bins))

        #supp = float(np.sum(arr_bin)) / float(n * (n - 1.0) / 2.0)
        #print(arr_bin)
        #print(len(arr_bin))
        #print("Support: " + str(supp))
        #temp_pos = Dataset.bin_rank(col_data, equal=self.equal)
        #supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)
        #print(temp_pos)
        #print(len(temp_pos))
        #print("Support: " + str(supp))

    def construct_bins_new(self, attr_data, seg_no):
        # execute binary rank to calculate support of pattern
        # valid_bins = list()  # numpy is very slow for append operations
        # n = self.attr_size
        n = self.row_count
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
                valid_bins.append(np.array([incr.tolist(), arr_pos], dtype=object))
            if arr_neg is None:
                invalid_bins.append(decr)
            else:
                valid_bins.append(np.array([decr.tolist(), arr_neg], dtype=object))
        print(valid_bins)
        self.valid_bins = np.array(valid_bins)
        self.invalid_bins = np.array(invalid_bins)
        if len(self.valid_bins) < 3:
            self.no_bins = True
        else:
            self.seg_count = self.valid_bins[0][1][0].size

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