import models
from util.data import CNFData

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import os
import torch

ns = models.NeuroSAT(iterations=3, hidden_dim=2)
cnf_data = CNFData("test.dimacs", "../../", split_literals=True)
cnf_data.loadForqes()
cnf_data.loadMUS()
features, mask = cnf_data.getFeatures()
data = Data(x=torch.tensor(features).float(), edge_index=torch.tensor(cnf_data.edge_index), mask=torch.tensor(mask),
                        edge_attr=torch.tensor(cnf_data.edge_attr).float(), y_mus=torch.tensor(cnf_data.mus_bin).float(),
                        y_sat=torch.tensor(cnf_data.sat).float())
train_loader = Batch.from_data_list([data,data,data])
print(torch.sigmoid(ns(train_loader)))

