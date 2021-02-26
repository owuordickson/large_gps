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

import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, file_path, chunks, min_sup, eq=False):
        self.csv_file = file_path
        self.thd_supp = min_sup
        self.equal = eq
        self.chunk_count = chunks
        self.titles = Dataset.read_csv_header(file_path)
        self.col_count = self.titles.size  # TO BE UPDATED
        self.row_count = 0  # TO BE UPDATED

    def init_gp_attributes(self):
        # print(self.print_header())
        self.read_csv_data()
        pass

    def read_csv_data(self):
        if self.titles.size > 0:
            df = pd.read_csv(self.csv_file, sep="[;,' '\t]", engine='python')
        else:
            df = pd.read_csv(self.csv_file, sep="[;,' '\t]", engine='python', header=None)
        print(df)

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
            header_row = pd.read_csv(file, sep="[;,' '\t]", engine='python', nrows=0).columns.tolist()

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
                return titles
        except Exception as error:
            print("Unable to read 1st line of CSV file")
            raise Exception("CSV file read error. " + str(error))
