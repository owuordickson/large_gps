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

    def __init__(self, file_path, min_sup=0, eq=False):
        data = Dataset.read_csv(file_path)
        if len(data) <= 1:
            self.data = np.array([])
            data = None
            print("csv file read error")
            raise Exception("Unable to read csv file or file has no data")
        else:
            print("Data fetched from csv file")
            self.thd_supp = min_sup
            self.equal = eq
            self.data = np.array([])
            self.titles = self.get_titles(data)
            self.cols_count = self.get_cols_count()
            self.data_size = self.get_size()
            self.time_cols = self.get_time_cols()
            self.attr_cols = self.get_attr_cols()
            # self.no_bins = False
            # self.attr_size = 0
            # self.step_name = ''
            # self.invalid_bins = np.array([])
            # self.valid_bins = np.array([])
            # data = None
            # self.init_attributes()

    def get_titles(self, data):
        # data = self.raw_data
        if data[0][0].replace('.', '', 1).isdigit() or data[0][0].isdigit():
            title = self.convert_data_to_array(data)
            return title
        else:
            if data[0][1].replace('.', '', 1).isdigit() or data[0][1].isdigit():
                title = self.convert_data_to_array(data)
                return title
            else:
                title = self.convert_data_to_array(data, has_title=True)
                return title

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