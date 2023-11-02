import torch
from torch_geometric.data import Data, Batch

import numpy as np


class Tester():
    def __init__(self, model_path, model_type):
        model_loaded = torch.load(model_path)

        self.model = model_type(iterations=model_loaded[1])
        self.model.load_state_dict(model_loaded[0])
    
    def getPred(self, cnf_data):
        features, mask = cnf_data.getFeatures()
        data_org = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.edge_index), mask=torch.tensor(mask),
                        edge_attr=torch.tensor(cnf_data.edge_attr).float())
        data = Batch.from_data_list([data_org])

        self.model.eval()
        with torch.no_grad():
            out = self.model(data, data.batch)

            out_final = torch.sigmoid(out[0]).numpy() * mask - (1-mask)
            return out_final