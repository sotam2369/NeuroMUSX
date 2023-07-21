import argparse
import os
import random

import numpy as np
from tqdm import tqdm
from pysat.solvers import Solver

def get_args():
    parser = argparse.ArgumentParser(description='Dataset Generator/Loader for NeuroMUSX')
    parser.add_argument('out_dir', action='store', type=str)
    parser.add_argument('n_pairs', action='store', type=int)

    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)

    parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)

    parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)

    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    return parser.parse_args()


def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")

def mk_out_filenames(args, n_vars, t):
    prefix = "sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d" % \
        (n_vars, args.p_k_2, args.p_geo, t)
    prefix = os.path.join(args.out_dir, prefix)
    return ("%s_sat=0.dimacs" % prefix, "%s_sat=1.dimacs" % prefix)

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

def gen_iclause_pair(args):
    n = random.randint(args.min_n, args.max_n)

    solver = Solver(name='m22')
    iclauses = []

    while True:
        k_base = 1 if random.random() < args.p_k_2 else 2
        k = k_base + np.random.geometric(args.p_geo)
        iclause = [int(i) for i in generate_k_iclause(n, k)]
        
        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat

if __name__ == '__main__':
    args = get_args()
    
    os.chdir("../src")

    if args.py_seed is not None: random.seed(args.py_seed)
    if args.np_seed is not None: np.random.seed(args.np_seed)

    for pair in tqdm(range(args.n_pairs)):
        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(args)
        out_filenames = mk_out_filenames(args, n_vars, pair)

        iclauses.append(iclause_unsat)
        write_dimacs_to(n_vars, iclauses, out_filenames[0])

        iclauses[-1] = iclause_sat
        write_dimacs_to(n_vars, iclauses, out_filenames[1])