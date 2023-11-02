import signal
import argparse
import os

import pickle
import numpy as np
from tqdm import tqdm
from util.data import CNFData

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--directory", type=str, default="../dataset/original/train")
    parser.add_argument("-out", "--output", type=str, default="../dataset/processed/train.pkl")
    parser.add_argument("-toopt", "--timeout-optimal", type=int, default=60, help="Timeout for optimal solver")
    parser.add_argument("-r", "--random", type=bool, default=True, help="Set to True if dataset is random")
    return parser.parse_args()

def raiseException(signum, frame):
    raise Exception()

if __name__ == "__main__":
    args = get_args()

    os.chdir('../src')

    output_list_unsat = []
    output_list_sat = []
    amt = 0
    split = 20
    listdir = os.listdir(args.directory)
    listdir_split = int(len(listdir) / split)
    print(listdir_split)

    for i in range(split):
        for cnf_file in tqdm(listdir[i*listdir_split:(i+1)*listdir_split]):
            if args.random and "sat=1" in cnf_file:
                cnf_data = CNFData(cnf_file, args.directory, sat=1)
                cnf_data.mus_bin = np.zeros(cnf_data.n_vars + cnf_data.n_clauses)
                output_list_sat.append(cnf_data)
                continue

            cnf_data = CNFData(cnf_file, args.directory, sat=0)
            cnf_data.loadForqes()


            signal.signal(signal.SIGALRM, cnf_data.s.killProcess)
            signal.alarm(args.timeout_optimal)

            try:
                cnf_data.loadMUS()
            except:
                cnf_data.loadMUS("musx")

            signal.alarm(0)
            cnf_data.deleteForqes()
            output_list_unsat.append(cnf_data)

        with open(args.output + ".part{}".format(i+1), "wb") as f:
            print("Saving...{0}/{1}".format(i+1, split))
            if args.random:
                pickle.dump([output_list_unsat, output_list_sat], f)
            else:
                pickle.dump(output_list_unsat, f)
            print("Saved")
        output_list_unsat = []
        output_list_sat = []