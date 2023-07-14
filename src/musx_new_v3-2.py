#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## musx.py
##
##  Created on: Jan 25, 2018
##      Author: Antonio Morgado, Alexey Ignatiev
##      E-mail: {ajmorgado, aignatiev}@ciencias.ulisboa.pt
##

"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        MUSX

    ==================
    Module description
    ==================

    This module implements a deletion-based algorithm [1]_ for extracting a
    *minimal unsatisfiable subset* (*MUS*) of a given (unsafistiable) CNF
    formula. This simplistic implementation can deal with *plain* and *partial*
    CNF formulas, e.g. formulas in the DIMACS CNF and WCNF formats.

    .. [1] Joao Marques-Silva. *Minimal Unsatisfiability: Models, Algorithms
        and Applications*. ISMVL 2010. pp. 9-14

    The following extraction procedure is implemented:

    .. code-block:: python

        # oracle: SAT solver (initialized)
        # assump: full set of assumptions

        i = 0

        while i < len(assump):
            to_test = assump[:i] + assump[(i + 1):]
            if oracle.solve(assumptions=to_test):
                i += 1
            else:
                assump = to_test

        return assump

    The implementation can be used as an executable (the list of available
    command-line options can be shown using ``musx.py -h``) in the following
    way:

    ::

        $ cat formula.wcnf
        p wcnf 3 6 4
        1 1 0
        1 2 0
        1 3 0
        4 -1 -2 0
        4 -1 -3 0
        4 -2 -3 0

        $ musx.py -s glucose3 -vv formula.wcnf
        c MUS approx: 1 2 0
        c testing clid: 0 -> sat (keeping 0)
        c testing clid: 1 -> sat (keeping 1)
        c nof soft: 3
        c MUS size: 2
        v 1 2 0
        c oracle time: 0.0001

    Alternatively, the algorithm can be accessed and invoked through the
    standard ``import`` interface of Python, e.g.

    .. code-block:: python

        >>> from pysat.examples.musx import MUSX
        >>> from pysat.formula import WCNF
        >>>
        >>> wcnf = WCNF(from_file='formula.wcnf')
        >>>
        >>> musx = MUSX(wcnf, verbosity=0)
        >>> musx.compute()  # compute a minimally unsatisfiable set of clauses
        [1, 2]

    Note that the implementation is able to compute only one MUS (MUS
    enumeration is not supported).

    ==============
    Module details
    ==============
"""

#
#==============================================================================
from __future__ import print_function
import getopt
import os
from pysat.formula import CNFPlus, WCNFPlus
from pysat.solvers import Solver, SolverNames
import re
import sys
from tester import Tester
import model
from CNF_Data import CNF_Data
import numpy as np
import time
import copy

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
            self.oracle2.delete()
            self.oracle = None
            self.oracle2 = None

    def compute(self, pred_org=None, only_pred=False):
        """
            This is the main method of the :class:`MUSX` class. It computes a
            set of soft clauses belonging to an MUS of the input formula.
            First, the method checks whether the formula is satisfiable. If it
            is, nothing else is done. Otherwise, an *unsatisfiable core* of the
            formula is extracted, which is later used as an over-approximation
            of an MUS refined in :func:`_compute`.
        """

        # cheking whether or not the formula is unsatisfiable
        #print("Coefficient:", coeff)
        #print(self.sels)
        #print(pred)
        #print("Starting compute")
        thres = -0.01
        next_assump = self.getList(thres, pred_org)
        approx = None
        smallest_approx = None
        changed = 0

        while not self.oracle.solve(assumptions=next_assump):

            approx_temp = self.oracle.get_core()
            #if approx and len(approx_temp) > len(approx):
            #    print("Stopped update:", len(approx_temp))
            #    break

            approx = approx_temp
            if not smallest_approx is None and len(approx) > len(smallest_approx)*4.8:
                print("Exitting due to size")
                break
            if not smallest_approx or len(approx) <= len(smallest_approx):
                smallest_approx = approx
                changed += 1

            print("Prediction update:", len(self.getList(thres, pred_org)), len(approx), thres)

            #print(self.oracle_time())
            if self.oracle_time() > 10:
                print("Exitting due to time")
                break

            thres = self.getSmallest(approx, pred_org)
            next_assump = self.getList(thres, pred_org)

        print(len(self.getList(thres, pred_org)))
        print("Theshold:", (thres))
        # Get numpy array corresponding to approx and sort by probability
        #print(approx)
        #self.oracle.solve(assumptions=self.sels)
        #smallest_approx = self.oracle.get_core()
        print("Size of Smallest Approximation:", len(smallest_approx))
        
        if self.verbose:
            print('c MUS approx:', ' '.join([str(self.vmap[sel] + 1) for sel in approx]), '0')
            
        print("Changed:", changed)
        if only_pred and changed < 2:
            print("Exitting due to no change")
            exit()
        #exit()
        self.deletion_start = self.oracle_time()
        # iterate over clauses in the approximation and try to delete them
        mus = self._compute(sorted(smallest_approx))

        self.deletion_fin = self.oracle_time()

        # return an MUS
        return list(map(lambda x: self.vmap[x] + 1, mus))
    
    def isContained(self, assump, approx):
        for var in approx:
            if not var in assump:
                return False
        return True

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

    def getMUSTime(self):
        return self.deletion_fin - self.deletion_start

    def _compute(self, approx):
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
            to_test = approx[:i] + approx[(i + 1):] # Removes 0, 1, 2 and so on
            sel, clid = approx[i], self.vmap[approx[i]]
            total += len(to_test)

            if self.verbose > 1:
                #print('c', approx[:i], approx[(i + 1):])
                print('c testing clid: {0}'.format(clid), end='')

                print('c Test size:', len(to_test))

            if self.oracle.solve(assumptions=to_test):
                if self.verbose > 1:
                    print(' -> sat (keeping {0})'.format(clid))

                i += 1
            else:
                if self.verbose > 1:
                    print(' -> unsat (removing {0})'.format(clid))

                approx = to_test
        #print(total)
        return approx


    def _compute_gnn(self, approx, approx_gnn):

        total = 0
        for i_index in range(len(approx_gnn)): # Add sorting mechanism
            i = approx.index(approx_gnn[i_index])

            to_test = approx[:i] + approx[(i + 1):] # Removes 0, 1, 2 and so on
            sel, clid = approx[i], self.vmap[approx[i]]
            total += len(to_test)

            if self.verbose > 1:
                #print('c', approx[:i], approx[(i + 1):])
                print('c testing clid: {0}'.format(clid), end='')
                print('c Test size:', len(to_test))

            if self.oracle.solve(assumptions=to_test):
                if self.verbose > 1:
                    print(' -> sat (keeping {0})'.format(clid))
            else:
                if self.verbose > 1:
                    print(' -> unsat (removing {0})'.format(clid))

                approx = to_test
            
        #print(total)
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
        opts, args = getopt.getopt(sys.argv[1:], 'hs:v', ['help', 'solver=', 'verbose', 'gnn'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    solver = 'm22'
    verbose = 0
    gnn = False

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt in ('-s', '--solver'):
            solver = str(arg)
        elif opt in ('-v', '--verbose'):
            verbose += 1
        elif opt in('--gnn'):
            gnn = True
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return solver, verbose, args, gnn


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


#
#==============================================================================
if __name__ == '__main__':
    solver, verbose, files, gnn = parse_options()
    only_pred = "-timeout" in files[0]
    if only_pred:
        print("TIMEOUT MODE ON")
    #tester = Tester('models_archive/model_3-40_satv2-3.pt', model.GNNSat_V2_3)
    if False:
        tester = Tester('models/model_1000.pt', model.GNNSat_V2_3)
        print("For withbit")
    elif "abstraction" in files[0]:
        tester = Tester('models_archive/model_ab3.pt', model.GNNSat_V2_3)
        #tester = Tester('models/model_100.pt', model.GNNSat_V2_3)
        print("For abstraction")
    elif "design" in files[0]:
        tester = Tester('models_archive/model_dd4.pt', model.GNNSat_V2_3)
        #tester = Tester('models/model_100.pt', model.GNNSat_V2_3)
        print("For design")
    elif "hardware" in files[0]:
        tester = Tester('models_archive/model_hv11.pt', model.GNNSat_V2_3)
        #tester = Tester('models/model_100.pt', model.GNNSat_V2_3)
        print("For hardware")
    elif "fdmus" in files[0]:
        #tester = Tester('models/model_100.pt', model.GNNSat_V2_3)
        tester = Tester('models_archive/model_fd2.pt', model.GNNSat_V2_3)
        print("For fd2")
    elif "application" in files[0]:
        tester = Tester('models_archive/model_app3.pt', model.GNNSat_V2_3)
        #tester = Tester('models/model_100.pt', model.GNNSat_V2_3)
        print("For application")
    else:
        tester = Tester('models_archive/model_withbit-7.pt', model.GNNSat_V2_3)


    if files:
        # parsing the input formula
        print("CNF File: ", files[0])
        if re.search('\.wcnf[p|+]?(\.(gz|bz2|lzma|xz))?$', files[0]):
            formula = WCNFPlus(from_file=files[0])
        else:  # expecting '*.cnf[,p,+].*'
            formula = CNFPlus(from_file=files[0]).weighted()
        print("Finished making CNF Module")
        cnf_data = CNF_Data(files[0], 'test_file')
        print("Finished making CNF Data")

        with MUSX(formula, solver=solver, verbosity=verbose) as musx:
            if gnn:
                if False:
                    temp = tester.test(cnf_data, cluster_coeff=True)
                    mus = musx.compute(prob_arr=temp[0], coeff=temp[1]*2)
                else:
                    gnn_pred = tester.test(cnf_data, cluster_coeff=False, round=True)
                    print("Finished GNN pred")
                    del tester, cnf_data
                    mus = musx.compute(pred_org=gnn_pred[0], only_pred=only_pred)
            else:
                mus = musx.compute(None)

            if mus:
                if verbose:
                    print('c nof soft: {0}'.format(len(formula.soft)))
                    print('c MUS size: {0}'.format(len(mus)))

                #print('v', ' '.join([str(clid) for clid in mus]), '0')
                print('c deletion time: {0:.4f}'.format(musx.getMUSTime()))
                print('c oracle time: {0:.4f}'.format(musx.oracle_time()))
                print('c MUS size: {0}'.format(len(mus)), '\n')