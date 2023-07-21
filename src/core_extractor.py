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

    for cnf_file in tqdm(os.listdir(args.directory)):
        if args.random and "sat=1" in cnf_file:
            cnf_data = CNFData(cnf_file, args.directory, sat=1)
            cnf_data.mus_bin = np.zeros(cnf_data.n_vars + cnf_data.n_clauses)
            output_list_sat.append(cnf_data)
            continue

        cnf_data = CNFData(cnf_file, args.directory, sat=0)

        signal.signal(signal.SIGALRM, raiseException)
        signal.alarm(args.timeout_optimal)

        try:
            cnf_data.loadMUS()
        except:
            cnf_data.loadMUS("musx")

        output_list_unsat.append(cnf_data)
    
    with open(args.output, "wb") as f:
        if args.random:
            pickle.dump([output_list_unsat, output_list_sat], f)
        else:
            pickle.dump(output_list_unsat, f)