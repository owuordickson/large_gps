# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@created: "03 May 2021"
Usage:
    $python init_acograd.py -f ../data/DATASET.csv -s 0.5
Description:
    f -> file path (CSV)
    s -> minimum support
"""

import sys
from optparse import OptionParser
import config as cfg
from pkg_algorithms import aco_grad_ch, aco_grad_h5


if __name__ == "__main__":
    if not sys.argv:
        algChoice = sys.argv[0]
        filePath = sys.argv[1]
        minSup = sys.argv[2]
        numCores = sys.argv[3]
        chkSize = sys.argv[4]
    else:
        optparser = OptionParser()
        optparser.add_option('-a', '--algorithmChoice',
                             dest='algChoice',
                             help='select algorithm',
                             default=cfg.ALGORITHM,
                             type='string')
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             default=cfg.DATASET,
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=cfg.MIN_SUPPORT,
                             type='float')
        optparser.add_option('-c', '--cores',
                             dest='numCores',
                             help='number of cores',
                             default=cfg.CPU_CORES,
                             type='int')
        optparser.add_option('-u', '--chunkSize',
                             dest='chkSize',
                             help='chunk size',
                             default=cfg.CHUNK_SIZE,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python3 main.py -a 'aco_ch' -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        algChoice = options.algChoice
        minSup = options.minSup
        numCores = options.numCores
        chkSize = options.chkSize

    import time
    import tracemalloc
    from pkg_algorithms.shared.profile import Profile

    if algChoice == 'aco_ch':
        # ACO-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = aco_grad_ch.GradACO.init_algorithm(filePath, minSup, numCores, chkSize)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_aco_v8' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    elif algChoice == 'aco_h5':
        # GA-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = aco_grad_h5.GradACO.init_algorithm(filePath, minSup, numCores, chkSize)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_aco_v7' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name)
        print(wr_text)
    else:
        print("Invalid Algorithm Choice!")