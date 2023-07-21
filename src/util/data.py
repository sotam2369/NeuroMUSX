import os

import numpy as np

from pysat.formula import CNFPlus
from pysat.examples.optux import OptUx
from pysat.examples.musx import MUSX

from util.solver_wrapper import Forqes

class CNFData():

    def __init__(self, file_name, file_dir, edge_features=[[[1,0], [1,0]], [[0,1], [0,1]]], literal_features=[[1,0], [0,1]], split_literals=False, sat=0):
        self.file_name = file_name
        self.file_dir = file_dir
        self.edge_features = edge_features
        self.literal_features = literal_features
        self.formula = None
        self.split_literals = split_literals
        self.sat = sat

        self.load()
    
    def load(self):
        if self.formula is None:
            self.formula = CNFPlus(from_file=os.path.join(self.file_dir, self.file_name)).weighted()
        self.n_vars = self.formula.nv
        self.n_clauses = len(self.formula.soft)

        edge_index = []
        edge_attr = []

        for clause in range(self.n_clauses):
            for literal in self.formula.soft[clause]:
                if literal < 0:
                    if self.split_literals:
                        edge_index.append([-literal - 1 + self.n_vars, self.n_vars*2 + clause])
                        edge_index.append([self.n_vars*2 + clause, -literal - 1 + self.n_vars])
                    else:
                        edge_attr.append(self.edge_features[1][0])
                        edge_attr.append(self.edge_features[1][1])
                        edge_index.append([-literal - 1, self.n_vars + clause])
                        edge_index.append([self.n_vars + clause, -literal - 1])
                else:
                    if self.split_literals:
                        edge_index.append([literal - 1, self.n_vars*2 + clause])
                        edge_index.append([self.n_vars*2 + clause, literal - 1])
                    else:
                        edge_attr.append(self.edge_features[0][0])
                        edge_attr.append(self.edge_features[0][1])
                        edge_index.append([literal - 1, self.n_vars + clause])
                        edge_index.append([self.n_vars + clause, literal - 1])
        
        self.edge_index = np.transpose(np.asarray(edge_index))
        self.edge_attr = np.asarray(edge_attr)

    
    def getFeatures(self):
        if self.split_literals:
            mask = np.concatenate((np.zeros(self.n_vars*2), np.ones(self.n_clauses)))
            lit_features = np.asarray([self.literal_features[0]]*(self.n_vars*2))
        else:
            mask = np.concatenate((np.zeros(self.n_vars), np.ones(self.n_clauses)))
            lit_features = np.asarray([self.literal_features[0]]*(self.n_vars))

        clause_features = np.asarray([self.literal_features[1]]*self.n_clauses)
        
        return np.concatenate((lit_features, clause_features)), mask

    def setSplitLiterals(self, split_literals):
        self.split_literals = split_literals
        self.load()

    def loadMUS(self, mode="forqes"):
        if mode == "forqes":
            s = Forqes()
            mus = s.solve(self, np.ones(self.n_clauses))
        elif mode == "optux":
            with OptUx(self.formula) as optux:
                mus = optux.compute()
        elif mode == "musx":
            with MUSX(self.formula) as musx:
                mus = musx.compute()
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        
        mus_bin = np.zeros(self.n_clauses)
        mus_bin[np.asarray(list(mus))-1] = 1
        self.mus = mus
        self.mus_bin = np.concatenate((np.zeros(self.n_vars), mus_bin))
    


