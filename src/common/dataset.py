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
        self.thd_supp = min_sup
        self.titles, self.data = Dataset.read_csv(file_path)
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self.get_time_cols()
        self.attr_cols = self.get_attr_cols()
        self.no_bins = False
        self.seg_count = 1
        # self.step_name = ''
        self.invalid_bins = np.array([])
        self.valid_bins = np.array([])
        # self.init_attributes()

    def get_attr_cols(self):
        all_cols = np.arange(self.col_count)
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

    def init_gp_attributes(self, seg_no, attr_data=None):
        # (check) implement parallel multiprocessing
        # 1. Transpose csv array data
        if attr_data is None:
            attr_data = self.data.T

        # 2. Construct and store 1-item_set valid bins
        valid_bins = list()
        invalid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')

            # 2a. Execute binary rank to calculate support of pattern
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

    @staticmethod
    def read_csv(file):
        # 1. Retrieve data set from file
        try:
            with open(file, 'r') as f:
                dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
                f.seek(0)
                reader = csv.reader(f, dialect)
                raw_data = list(reader)
                f.close()

            if len(raw_data) <= 1:
                print("Unable to read CSV file")
                raise Exception("CSV file read error. File has little or no data")
            else:
                print("Data fetched from CSV file")
                # 2. Get table headers
                if raw_data[0][0].replace('.', '', 1).isdigit() or raw_data[0][0].isdigit():
                    titles = np.array([])
                else:
                    if raw_data[0][1].replace('.', '', 1).isdigit() or raw_data[0][1].isdigit():
                        titles = np.array([])
                    else:
                        # titles = self.convert_data_to_array(data, has_title=True)
                        keys = np.arange(len(raw_data[0]))
                        values = np.array(raw_data[0], dtype='S')
                        titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
                        raw_data = np.delete(raw_data, 0, 0)
                return titles, np.asarray(raw_data)
                # return Dataset.get_tbl_headers(temp)
        except Exception as error:
            print("Unable to read CSV file")
            raise Exception("CSV file read error. " + str(error))

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
