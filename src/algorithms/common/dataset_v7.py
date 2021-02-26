# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Anne Laurent"
@license: "MIT"
@version: "6.0"
@email: "owuordickson@gmail.com"
@created: "26 Feb 2021"
@modified: "26 Feb 2021"

Changes
-------
1. Fetch all binaries during initialization
2. Replaced loops for fetching binary rank with numpy function
3. Chunks CSV file read

"""
import csv
import gc
from dateutil.parser import parse
import time
import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, file_path, c_size, min_sup, eq=False):
        self.csv_file = file_path
        self.thd_supp = min_sup
        self.equal = eq
        self.chunk_size = c_size
        self.titles, self.col_count, self.time_cols = Dataset.read_csv_header(file_path)
        self.attr_cols = self.get_attr_cols()
        self.row_count = 0  # TO BE UPDATED

    def get_attr_cols(self):
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def init_gp_attributes(self):
        # print(self.print_header())
        # print(self.attr_cols)
        n = self.row_count
        valid_bins = list()
        for col in self.attr_cols:
            self.read_csv_data(col)

    def read_csv_data(self, col):
        if self.titles.dtype is np.int32:
            df = pd.read_csv(self.csv_file, sep="[;,' '\t]", header=None, engine='python')
        else:
            df = pd.read_csv(self.csv_file, sep="[;,' '\t]", usecols=[col], engine='python')
        print(df)
        return df.values

    def print_header(self):
        str_header = "Header Columns/Attributes\n-------------------------\n"
        for txt in self.titles:
            try:
                str_header += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                try:
                    str_header += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')
                except IndexError:
                    str_header += (str(txt) + '\n')
        return str_header

    @staticmethod
    def read_csv_header(file):
        try:
            df = pd.read_csv(file, sep="[;,' '\t]", engine='python', nrows=1)
            header_row = df.columns.tolist()

            if len(header_row) <= 0:
                print("CSV file is empty!")
                raise Exception("CSV file read error. File has little or no data")
            else:
                print("Header titles fetched from CSV file")
                # 2. Get table headers
                if header_row[0].replace('.', '', 1).isdigit() or header_row[0].isdigit():
                    titles = np.arange(len(header_row))
                else:
                    if header_row[1].replace('.', '', 1).isdigit() or header_row[1].isdigit():
                        titles = np.arange(len(header_row))
                    else:
                        # titles = self.convert_data_to_array(data, has_title=True)
                        keys = np.arange(len(header_row))
                        values = np.array(header_row, dtype='S')
                        titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
                del header_row
                gc.collect()
                return titles, titles.size, Dataset.get_time_cols(df.values)
        except Exception as error:
            print("Unable to read 1st line of CSV file")
            raise Exception("CSV file read error. " + str(error))

    @staticmethod
    def get_time_cols(data):
        # Retrieve first column only
        time_cols = list()
        # n = len(data)
        for i in range(data.shape[1]):  # check every column/attribute for time format
            row_data = str(data[0][i])
            try:
                time_ok, t_stamp = Dataset.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

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
