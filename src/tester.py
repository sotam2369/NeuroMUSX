import torch
import pickle
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np

import model

class Tester():

    def __init__(self, model_path, model_type):
        model_loaded = torch.load(model_path)
        
        self.model = model_type(iterations = model_loaded[1])
        self.model.load_state_dict(model_loaded[0])
    
    def test(self, cnf_data, cluster_coeff=False, round=False):
        if cluster_coeff:
            cnf_data.loadDimacs(loadNetworkx=True)
        features, mask = cnf_data.getFeatures()
        data_org = Data(x=features, edge_index=torch.tensor(cnf_data.out_graph), mask=torch.tensor(mask), edge_attr = torch.tensor(cnf_data.edge_attr).float(),cnf_data=cnf_data, vars=cnf_data.getVars())
        data = Batch.from_data_list([data_org])

        self.model.eval()
        with torch.no_grad():
            out = self.model(data)

            out = torch.sigmoid(out).numpy()
            out = np.where(out == 0.5, -1, out)
            if round:
                out_final = np.where(out > 0.5, 1, 0)
                out_final = np.argwhere(out_final == 1).flatten() + 1
                return out, out_final.tolist()
            else:
                out = out.tolist()
                if cluster_coeff:
                    return out, nx.algorithms.approximation.average_clustering(cnf_data.networkx_graph)
                else:
                    return out
        

if __name__ == '__main__':
    tester = Tester('models/model_1000.pt', model.GNNSat_V2)

    with open('extracted_cores/unsat_cores_train_40.pickle', 'rb') as handle:
        unsat_cores_train = pickle.load(handle)
    print(list(unsat_cores_train.keys())[0])
    print(tester.test(unsat_cores_train[list(unsat_cores_train.keys())[0]]))