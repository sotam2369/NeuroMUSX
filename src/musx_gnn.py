#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## musx.py
##
##  Created on: Jan 25, 2018
##      Author: Antonio Morgado, Alexey Ignatiev
##      E-mail: {ajmorgado, aignatiev}@ciencias.ulisboa.pt
##
##  Edited on: July 21, 2023
##      Author: Sota Moriyama
##
##
#==============================================================================
from __future__ import print_function
import getopt
import os
from pysat.formula import CNFPlus, WCNFPlus
from pysat.solvers import Solver
import re
import sys
from tester import Tester
from models import NeuroMUSX, NeuroMUSX_V2
from util.data import CNFData
from musx_org import MUSX as MUSX_Org
import numpy as np
import csv

#
#==============================================================================
class MUSX(object):
    """
        MUS eXtractor using the deletion-based algorithm. The algorithm is
        described in [1]_ (also see the module description above). Essentially,
        the algorithm can be seen as an iterative process, which tries to
        remove one soft clause at a time and check whether the remaining set of
        soft clauses is still unsatisfiable together with the hard clauses.

        The constructor of :class:`MUSX` objects receives a target
        :class:`.WCNF` formula, a SAT solver name, and a verbosity level. Note
        that the default SAT solver is MiniSat22 (referred to as ``'m22'``, see
        :class:`.SolverNames` for details). The default verbosity level is
        ``1``.

        :param formula: input WCNF formula
        :param solver: name of SAT solver
        :param verbosity: verbosity level

        :type formula: :class:`.WCNF`
        :type solver: str
        :type verbosity: int
    """

    def __init__(self, formula, solver='m22', verbosity=1):
        """
            Constructor.
        """

        topv, self.verbose = formula.nv, verbosity
        self.org_topv = formula.nv
        self.clause_amt = len(formula.soft)

        # clause selectors and a mapping from selectors to clause ids
        self.sels, self.vmap = [], {}

        # constructing the oracle
        self.oracle = Solver(name=solver, bootstrap_with=formula.hard,
                use_timer=True)

        if isinstance(formula, WCNFPlus) and formula.atms:
            assert self.oracle.supports_atmost(), \
                    '{0} does not support native cardinality constraints. Make sure you use the right type of formula.'.format(solver)

            for atm in formula.atms:
                self.oracle.add_atmost(*atm)

        # relaxing soft clauses and adding them to the oracle
        for i, cl in enumerate(formula.soft):
            topv += 1

            self.sels.append(topv)
            self.vmap[topv] = i

            self.oracle.add_clause(cl + [-topv])

    def __enter__(self):
        """
            'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """

        self.oracle.delete()
        self.oracle = None

    def delete(self):
        """
            Explicit destructor of the internal SAT oracle.
        """

        if self.oracle:
            self.oracle.delete()
            self.oracle = None

    def compute(self, pred=None, timeout=1000):
        thres = -0.01
        next_assump = self.getList(thres, pred)
        
        approx = None
        smallest_approx = None

        changed = 0

        while not self.oracle.solve(assumptions=next_assump):
            approx_temp = self.oracle.get_core()

            approx = approx_temp
            if not smallest_approx is None and len(approx) > len(smallest_approx)*4.8:
                print("c Exitting due to size")
                break
            if not smallest_approx or len(approx) <= len(smallest_approx):
                smallest_approx = approx
                changed += 1

            print("c Prediction update:", len(self.getList(thres, pred)), len(approx), thres)

            if self.oracle_time() > 10:
                print("c Exitting due to time")
                break

            thres = self.getSmallest(approx, pred)
            next_assump = self.getList(thres, pred)

        print("c Theshold:", (thres))
        print("c Size of Smallest Approximation:", len(smallest_approx))
        
        if self.verbose:
            print('c MUS approx:', ' '.join([str(self.vmap[sel] + 1) for sel in approx]), '0')
            
        print("c Changed:", changed)
        mus = self._compute(sorted(smallest_approx), timeout=timeout)

        return list(map(lambda x: self.vmap[x] + 1, mus))

    def getList(self, threshold, pred):
        pred = np.where(pred > threshold, 1, 0)
        pred = np.argwhere(pred == 1).flatten() + 1
        return pred.tolist()

    def getSmallest(self, approx, pred):
        smallest = 1
        for var in approx:
            if pred[var-1] < smallest:
                smallest = pred[var-1]
        return smallest

    def _compute(self, approx, timeout=1000):
        """
            Deletion-based MUS extraction. Given an over-approximation of an
            MUS, i.e. an unsatisfiable core previously returned by a SAT
            oracle, the method represents a loop, which at each iteration
            removes a clause from the core and checks whether the remaining
            clauses of the approximation are unsatisfiable together with the
            hard clauses.

            Soft clauses are (de)activated using the standard MiniSat-like
            assumptions interface [2]_. Each soft clause :math:`c` is augmented
            with a selector literal :math:`s`, e.g. :math:`(c) \gets (c \\vee
            \\neg{s})`. As a result, clause :math:`c` can be activated by
            assuming literal :math:`s`. The over-approximation provided as an
            input is specified as a list of selector literals for clauses in
            the unsatisfiable core.

            .. [2] Niklas Eén, Niklas Sörensson. *Temporal induction by
                incremental SAT solving*. Electr. Notes Theor. Comput. Sci.
                89(4). 2003. pp. 543-560

            :param approx: an over-approximation of an MUS
            :type approx: list(int)

            Note that the method does not return. Instead, after its execution,
            the input over-approximation is refined and contains an MUS.
        """

        i = 0
        total = 0
        while i < len(approx): # Add sorting mechanism
            if i % 10 == 0 and self.oracle_time() > timeout:
                print("c Timeout limit exceeded")
                break
            to_test = approx[:i] + approx[(i + 1):] # Removes 0, 1, 2 and so on
            sel, clid = approx[i], self.vmap[approx[i]]
            total += len(to_test)

            if self.verbose > 1:
                print('c testing clid: {0}'.format(clid), end='')

            if self.oracle.solve(assumptions=to_test):
                if self.verbose > 1:
                    print(' -> sat (keeping {0})'.format(clid))

                i += 1
            else:
                if self.verbose > 1:
                    print(' -> unsat (removing {0})'.format(clid))

                approx = to_test
        return approx

    def oracle_time(self):
        """
            Method for calculating and reporting the total SAT solving time.
        """

        return self.oracle.time_accum()


#
#==============================================================================
def parse_options():
    """
        Parses command-line option
        """
    try:
        opts, args = getopt.getopt(sys.argv[2:], 'hs:vm:o:', ['help', 'solver=', 'verbose', 'model=', 'output='])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    solver = 'm22'
    verbose = 0
    gnn = "../pretrained_models/model_random.pt"
    output = '../log/output.csv'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-s', '--solver'):
            solver = str(arg)
        elif opt in ('-v', '--verbose'):
            verbose += 1
        elif opt in('-m', '--model'):
            gnn = "../pretrained_models/" + str(arg)
        elif opt in('-o', '--output'):
            output = '../log/' + str(arg)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)
    return solver, verbose, sys.argv[1], gnn, output


#
#==============================================================================
def usage():
    """
        Prints usage message.
        """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] dimacs-file')
    print('Options:')
    print('        -h, --help')
    print('        -s, --solver     SAT solver to use')
    print('                         Available values: g3, lgl, mcb, mcm, mpl, m22, mc, mgh (default: m22)')
    print('        -v, --verbose    Be verbose')
    print('        --output         Name of output file csv')
    print('        --model          Name of model to use')


#
#==============================================================================
if __name__ == '__main__':
    os.chdir("../src")
    solver, verbose, files, gnn, output_file = parse_options()
    tester = Tester(gnn, NeuroMUSX_V2)
    check_both = True
    TIMEOUT = 1000


    if files:
        print("c CNF File: ", files)
        if re.search('\.wcnf[p|+]?(\.(gz|bz2|lzma|xz))?$', files):
            formula = WCNFPlus(from_file=files)
        else:
            formula = CNFPlus(from_file=files).weighted()
        print("c Finished making CNF Module")
        cnf_data = CNFData(files, '')
        print("c Finished making CNF Data")

        final_output = [files]
        with MUSX_Org(formula, solver=solver, verbosity=verbose) as musx:
            mus = musx.compute(timeout=TIMEOUT)
            if mus:
                if verbose:
                    print('c nof soft (normal): {0}'.format(len(formula.soft)))
                    print('c MUS size (normal): {0}'.format(len(mus)))

                print('c oracle time (normal): {0:.4f}'.format(musx.oracle_time()))
                print('c MUS size (normal): {0}'.format(len(mus)), '\n')
                final_output.append(len(mus))
                final_output.append(min(musx.oracle_time(), TIMEOUT))

        with MUSX(formula, solver=solver, verbosity=verbose) as musx:
            gnn_pred = tester.getPred(cnf_data)
            mus = musx.compute(pred=gnn_pred, timeout=TIMEOUT)

            if mus:
                if verbose:
                    print('c nof soft (gnn): {0}'.format(len(formula.soft)))
                    print('c MUS size (gnn): {0}'.format(len(mus)))

                print('c oracle time (gnn): {0:.4f}'.format(musx.oracle_time()))
                print('c MUS size (gnn): {0}'.format(len(mus)), '\n')
                final_output.append(len(mus))
                final_output.append(min(musx.oracle_time(), TIMEOUT))

                if not os.path.exists(output_file):
                    with open(output_file, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(["File", "Normal MUS Size", "Normal Oracle Time", "GNN MUS Size", "GNN Oracle Time"])
                        csv_writer.writerow(final_output)
                else:
                    with open(output_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(final_output)
                