import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import z3
from pysat.examples.optux import OptUx
from pysat.examples.musx import MUSX
from pysat.formula import CNF
import solver
import time
from tqdm import tqdm
import torch
import itertools
import networkx as nx
import musgraph_editor
from pysat.solvers import Solver
from pysat.formula import CNFPlus

class CNF_Data():

    def __init__(self, prob_file, name, edge_feature=[[[1,0],[0,1]],[[-1,0],[0,-1]]], neurosat=False):
        self.prob_file = prob_file
        self.edge_feature = edge_feature
        self.name = name
        self.loadDimacs(neuro_sat=neurosat)

    def displayGraph(self, large=False):
        G = nx.Graph()

        x_list = []
        for var in range(self.n_var):
            G.add_node("x" + str(var+1), bipartite=0)
            G.add_node("-x" + str(var+1), bipartite=0)
            x_list.append("x" + str(var+1))
            x_list.append("-x" + str(var+1))

        clause_list = []
        for clause in range(self.n_clause):
            G.add_node("c" + str(clause+1), bipartite=1)
            clause_list.append("c" + str(clause+1))
            for var in self.clauses[clause]:
                if int(var) < 0:
                    G.add_edge("c" + str(clause+1), "-x" + str(-int(var)))
                elif int(var) > 0:
                    G.add_edge("c" + str(clause+1), "x" + var)

        pos = dict()
        pos.update( (n, (1, i)) for i, n in enumerate(x_list) )
        pos.update( (n, (2, i)) for i, n in enumerate(clause_list) )

        if large:
            plt.figure(figsize=(60,120))
            nx.draw_networkx(G, pos=pos, node_size=0.1, width=0.4)
        else:
            nx.draw_networkx(G, pos=pos)

        plt.savefig("output_image/graph_" + self.name + ".png")
        plt.clf()
    

    def loadDimacs(self, loadNetworkx=False, neuro_sat=False, loadSolver=False):
        with open(self.prob_file, 'r') as cnf_file:
            lines = cnf_file.readlines()
            clauses = [line.strip().split()[:-1] for line in lines if line[0] not in ['c', 'p'] and line.strip()] # 0 is removed here

        n_clause = len(clauses)
        n_var = max([max([abs(int(var)) for var in clause]) for clause in clauses])

        out_graph = []
        edge_attr = []

        if loadSolver:
            formula = CNFPlus(from_file=self.prob_file).weighted()
            self.oracle = Solver(name='m22', bootstrap_with=formula.hard,
                    use_timer=True)
            topv = n_var
            for i, cl in enumerate(formula.soft):
                topv += 1

                self.oracle.add_clause(cl + [-topv])
        
        for clause in range(n_clause):
            for var in clauses[clause]:
                if int(var) < 0:
                    if neuro_sat:
                        out_graph.append([clause+n_var*2, -int(var) - 1 + n_var])
                        out_graph.append([-int(var) - 1 + n_var, clause+n_var*2])
                    else:
                        out_graph.append([clause+n_var, -int(var) - 1])
                        out_graph.append([-int(var) - 1, clause+n_var])
                        edge_attr.append(self.edge_feature[1][0])
                        edge_attr.append(self.edge_feature[1][1])
                else:
                    if neuro_sat:
                        out_graph.append([clause+n_var*2, int(var) - 1])
                        out_graph.append([int(var) - 1, clause+n_var*2])
                    else:
                        out_graph.append([clause+n_var, int(var) - 1])
                        out_graph.append([int(var) - 1, clause+n_var])
                        edge_attr.append(self.edge_feature[0][0])
                        edge_attr.append(self.edge_feature[0][1])
        self.n_var = n_var
        self.n_clause = n_clause
        self.clauses = clauses
        self.edge_attr = np.asarray(edge_attr)
        self.out_graph = np.transpose(np.asarray(out_graph))
        if loadNetworkx:
            networkx_graph = []
            for clause in range(n_clause):
                connected = self.out_graph[0][self.out_graph[1]==clause]
                for edge in itertools.combinations(sorted(connected.tolist()), 2):
                    if edge not in networkx_graph:
                        networkx_graph.append(edge)
            self.networkx_graph = nx.Graph(networkx_graph)
    

    def getFeatures(self, use_vars = False, neuro_sat = False):
        if use_vars:
            mask = np.ones(self.n_var + self.n_clause)
        elif neuro_sat:
            mask = np.concatenate((np.zeros(self.n_var*2), np.ones(self.n_clause)))
        else:
            mask = np.concatenate((np.zeros(self.n_var), np.ones(self.n_clause)))
        if neuro_sat:
            var_features = np.asarray([[1,0]]*(self.n_var*2))
        else:
            var_features = np.asarray([[1,0]]*self.n_var)
        clause_features = np.repeat(np.asarray([[0,1]]), self.n_clause, axis=0)
        return torch.tensor(np.concatenate((var_features, clause_features))).float(), mask

    def loadUnsatVariables(self):
        for clause in range(len(self.clauses)):
            if self.unsat_cores[0][self.n_var + clause] == 1:
                for var in self.clauses[clause]:
                    self.unsat_cores[0][abs(int(var))-1] = 1

    def getVars(self):
        return torch.tensor(np.concatenate((np.ones(self.n_var), np.zeros(self.n_clause)))).float()

    def isUnsatCoreZ3(self, pred):
        sum_pred = np.sum(pred)
        sum_unsat_core = np.sum(self.unsat_core)
        if sum_pred == self.n_clause or sum_pred == 0 or sum_pred > sum_unsat_core * 2:
            return False
        s = z3.Solver()
        pred_args = (np.argwhere(pred == 1)-self.n_var).tolist()

        bools = []
        for i in range(self.n_var):
            bools.append(z3.Bool('x' + str(i)))

        for clause in pred_args:
            clause_z3 = []
            for var in self.clauses[clause[0]]:
                var = int(var)
                if var > 0:
                    clause_z3.append(bools[var-1])
                elif var < 0:
                    clause_z3.append(z3.Not(bools[-var-1]))
            s.add(z3.Or(clause_z3))
            
        return s.check() == z3.unsat

    def isUnsatCoreKissat(self, pred, output=True):
        sum_pred = np.sum(pred)
        #sum_unsat_core = np.sum(self.unsat_core)
        if sum_pred == self.n_clause or sum_pred == 0: #or sum_pred > sum_unsat_core * 2:
            return False
        for unsat_core in self.unsat_cores:
            if np.min(pred - unsat_core) == 0:
                return True
        s = solver.KissatSolver()
        s_res = s.solve(self, pred[self.n_var:])
        if s_res == 0 and output:
            s.outputCoreDRAT()
        return s_res == 0

    def isUnsatAssump(self, pred):
        sum_pred = np.sum(pred)
        if sum_pred == self.n_clause or sum_pred == 0:
            return False
        assump = np.argwhere(pred == 1) + 1
        if len(assump) > 1:
            assump = np.squeeze(assump)
        else:
            assump = assump[0]
        
        #isUnsat = not self.oracle.solve(assumptions=(assump).tolist())
        return not self.oracle.solve(assumptions=(assump).tolist())


    
    def isMUS(self, pred):
        s = solver.KissatSolver()
        for clause in np.argwhere(pred[self.n_var:] == 1):
            pred[self.n_var + clause] = 0
            if s.solve(self, pred[self.n_var:]) == 0:
                return False
            pred[self.n_var + clause] = 1
        return True

    def loadUnsatCoreZ3(self):
        s = z3.Solver()
        s.set(unsat_core=True)

        bools = []
        for i in range(self.n_var):
            bools.append(z3.Bool('x' + str(i)))


        for clause in range(self.n_clause):
            clause_z3 = []
            for var in self.clauses[clause]:
                var = int(var)
                if var > 0:
                    clause_z3.append(bools[var-1])
                elif var < 0:
                    clause_z3.append(z3.Not(bools[-var-1]))
            s.assert_and_track(z3.Or(clause_z3), str(clause))
        
        s_res = s.check()
        if s_res == z3.unsat:
            c = s.unsat_core()
            prob_list = np.asarray([int(expr.sexpr().replace("|","")) for expr in c]) + self.n_var
        
            unsat_core = np.zeros(self.n_clause + self.n_var)
            unsat_core[prob_list] = 1

            self.unsat_core = unsat_core
        elif s_res == z3.sat:
            self.unsat_core = np.zeros(self.n_clause + self.n_var)
        else:
            print(s_res)
    

    def loadUnsatCoreCustom(self, checkUnsat=False):
        s = solver.KissatSolver()
        
        s_res = 0
        if checkUnsat:
            s_res = s.solve(self, np.ones(self.n_clause))
        else:
            s.createPredFile(self, np.ones(self.n_clause))
        if s_res == 0:
            s.outputCoreMUSer2()
            with open("./temp_output/unsat_core.cnf", 'r') as cnf_file:
                lines = cnf_file.readlines()
                clauses = [sorted(line.strip().split()[:-1]) for line in lines if line[0] not in ['c', 'p'] and line.strip()]
            unsat_core = []
            for clause in self.clauses:
                if clause in clauses:
                    clauses.remove(clause)
                    unsat_core.append(1)
                else:
                    unsat_core.append(0)

            self.unsat_core = np.concatenate((np.zeros(self.n_var), np.asarray(unsat_core)))
            #print(np.sum(self.unsat_core)/float(self.n_clause), "%")
        else:
            print("SAT")


    def loadUnsatCoreAll(self):
        cnf = CNF(from_file=self.prob_file).weighted()
        self.unsat_cores = []

        with OptUx(cnf) as optux:
            i = 0
            for mus in optux.enumerate():
                clause_bin = np.zeros(self.n_clause)
                clause_bin[np.asarray(list(mus))-1] = 1
                self.unsat_cores.append(np.concatenate((np.zeros(self.n_var), clause_bin)))
                i += 1
                if i == 10000:
                    break
    


    def loadMinimalMUS(self):
        cnf = CNF(from_file=self.prob_file).weighted()
        self.unsat_cores = []

        with OptUx(cnf) as optux:
            mus = optux.compute()
            clause_bin = np.zeros(self.n_clause)
            clause_bin[np.asarray(list(mus))-1] = 1
            self.unsat_cores.append(np.concatenate((np.zeros(self.n_var), clause_bin)))
    
    def loadMinimalMUSForques(self):
        s = solver.Forques()

        mus = s.solve(self, np.ones(self.n_clause))
        #print(mus)
        self.unsat_cores = []
        clause_bin = np.zeros(self.n_clause)
        clause_bin[np.asarray(list(mus))-1] = 1
        self.unsat_cores.append(np.concatenate((np.zeros(self.n_var), clause_bin)))

    def loadMUS(self):
        cnf = CNF(from_file=self.prob_file).weighted()
        self.unsat_cores = []

        with MUSX(cnf) as musx:
            mus = musx.compute()
            clause_bin = np.zeros(self.n_clause)
            clause_bin[np.asarray(list(mus))-1] = 1
            self.unsat_cores.append(np.concatenate((np.zeros(self.n_var), clause_bin)))
    
    def getClosestCore(self, pred ,print_data=False):
        if len(self.unsat_cores) == 1:
            return self.unsat_cores[0]
        #return self.unsat_cores[0]
        min_dist = self.n_clause
        closest_core = 0
        sum_pred = np.sum(pred)
        if sum_pred != self.n_clause and sum_pred != 0:
            i = 0

            for core in self.unsat_cores:
                dist = np.sum(np.abs(pred-core)) # Calculate distance between two vectors
                if dist < min_dist:
                    min_dist = dist
                    closest_core = int(i)
                i += 1
        if print_data:
            print("\n\nCore distance")
            print("Minimum distance: ", min_dist)
            print("Closest core: ", closest_core, "out of", len(self.unsat_cores))
            print(self.unsat_cores[closest_core][-100:])
            print(pred[-100:])
            print("\n\n")
        return self.unsat_cores[closest_core]

    def outputPrediction(self, pred, file_name):
        with open("./temp_output/" + file_name, "w+") as pred_core:
            s = solver.Solver()
            musgraph_editor.writeGraph(pred_core, np.argwhere(pred[self.n_var:]==1)+1, CNF(from_file=self.prob_file).weighted())
            #pred_core.writelines(s.getCNF(self, pred[self.n_var:]))