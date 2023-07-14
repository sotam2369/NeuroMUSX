import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
from torch_geometric.utils import add_self_loops

class GNNSat(nn.Module):
    def __init__(self, iterations=1):
        super(GNNSat, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.TransformerConv(2, 64, edge_dim=2)
        self.gcn2 = gnn.TransformerConv(64, 64, edge_dim=2)
        self.final = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        # Layer normalization
        self.ln = gnn.BatchNorm(64)
        self.relu = nn.LeakyReLU()
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.relu(self.ln(self.gcn(x, edge_index, edge_attr)))
        x_out = self.relu(self.gcn2(x_out, edge_index, edge_attr))

        for _ in range(self.iterations-1):
            x_out = self.relu(self.gcn2(x_out, edge_index, edge_attr))
            if print_data:
                print(self.final(x_out)[-3:])

        x_out = torch.squeeze(self.final(x_out))
        if print_data:
            print(x_out[-3:])
        
        x_out = x_out * data.mask
        return x_out



class GNNSat_V2(nn.Module):
    def __init__(self, iterations=1, use_mask=True):
        super(GNNSat_V2, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.GATConv(2, 64, edge_dim=2)
        self.gcn2 = gnn.GATConv(64, 64, edge_dim=2)
        self.final = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        # Layer normalization
        self.ln = gnn.BatchNorm(64)
        self.relu = nn.LeakyReLU()
        self.use_mask = use_mask
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.relu(self.ln(self.gcn(x, edge_index, edge_attr)))
        x_out = self.relu(self.gcn2(x_out, edge_index, edge_attr))
        sbs = []

        for i in range(self.iterations-1):
            x_out = self.relu(self.gcn2(x_out, edge_index, edge_attr))
            if print_data:
                print(self.final(x_out)[-3:])
                temp = torch.sigmoid(torch.squeeze(self.final(x_out)) * data.mask).detach().to('cpu').numpy()
                sbs.append(temp)
                temp = np.where(temp > 0.5, 1, 0)
                data.cnf_data[0].outputPrediction(temp, str(i) + ".graphml")

        x_out = torch.squeeze(self.final(x_out))
        if print_data:
            print(x_out[-3:])
            print(sbs)
        
        if self.use_mask:
            x_out = x_out * data.mask
        return x_out



class GNNSat_V2_2(nn.Module):
    def __init__(self, iterations=1):
        super(GNNSat_V2_2, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.GATConv(2, 64, edge_dim=2)
        self.gcn2 = gnn.GATConv(64, 64, edge_dim=2)
        self.gcn_lit = gnn.GATConv(2, 64, edge_dim=2)
        self.gcn2_lit = gnn.GATConv(64, 64, edge_dim=2)
        self.final = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        # Layer normalization
        self.ln = gnn.BatchNorm(64)
        self.relu = nn.LeakyReLU()
    
    def forward_Lit(self, gcn, x, edge_index, edge_attr, mask):
        return gcn(x, edge_index, edge_attr)*(1-mask)

    def forward_Clause(self, gcn, x, edge_index, edge_attr, mask):
        return gcn(x, edge_index, edge_attr)*mask

    def forward_GNN(self, gcn_clause, gcn_lit, x, edge_index, edge_attr, mask):
        return self.forward_Lit(gcn_lit, x, edge_index, edge_attr, mask) + self.forward_Clause(gcn_clause, x, edge_index, edge_attr, mask)
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        mask = torch.unsqueeze(data.mask, -1).float()

        x_out = self.relu(self.ln(self.forward_GNN(self.gcn, self.gcn_lit, x, edge_index, edge_attr, mask)))
        x_out = self.relu(self.forward_GNN(self.gcn2, self.gcn2_lit, x_out, edge_index, edge_attr, mask))
        sbs = []

        for i in range(self.iterations-1):
            x_out = self.relu(self.forward_GNN(self.gcn2, self.gcn2_lit, x_out, edge_index, edge_attr, mask))
            if print_data:
                print(self.final(x_out)[-3:])
                temp = torch.sigmoid(torch.squeeze(self.final(x_out)) * data.mask).detach().to('cpu').numpy()
                sbs.append(temp)
                temp = np.where(temp > 0.5, 1, 0)
                data.cnf_data[0].outputPrediction(temp, str(i) + ".graphml")

        x_out = torch.squeeze(self.final(x_out))
        if print_data:
            print(x_out[-3:])
            print(sbs)
        
        x_out = x_out * data.mask
        return x_out

class GNNSat_V3(nn.Module):
    def __init__(self, iterations=1):
        super(GNNSat_V3, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.GATConv(2, 64, edge_dim=2)
        gcn_main = []
        for _ in range(iterations):
            gcn_main.append(gnn.GATConv(64, 64, edge_dim=2))
        self.gcn_main = nn.Sequential(*gcn_main)
        self.final = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        # Layer normalization
        self.ln = gnn.BatchNorm(64)
        self.relu = nn.LeakyReLU()
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.relu(self.ln(self.gcn(x, edge_index, edge_attr)))
        x_out = self.relu(self.gcn_main[0](x_out, edge_index, edge_attr))

        for i in range(1, self.iterations):
            x_out = self.relu(self.gcn_main[i](x_out, edge_index, edge_attr))
            if print_data:
                print(self.final(x_out)[-3:])
                data.cnf_data[0].outputPrediction(torch.sigmoid(torch.squeeze(self.final(x_out))).detach().to('cpu').numpy(), str(i) + ".cnf")

        x_out = torch.squeeze(self.final(x_out))
        if print_data:
            print(x_out[-3:])
            data.cnf_data[0].outputPrediction(x_out, "final.cnf")
        
        x_out = x_out * data.mask
        return x_out


class GNNSat_V2_3(nn.Module):
    def __init__(self, iterations=1, use_mask=True, bit_supervision=False):
        super(GNNSat_V2_3, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.GATv2Conv(2, 8, edge_dim=2, heads=16)
        gcn_main = []
        for _ in range(iterations):
            gcn_main.append(gnn.GATv2Conv(128, 8, edge_dim=2, heads=16))
        self.gcn_main = nn.Sequential(*gcn_main)
        self.final = gnn.GATv2Conv(128, 1, edge_dim=2, heads=16, concat=False)
        #self.final = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        # Layer normalization
        self.relu = nn.LeakyReLU()
        self.use_mask = use_mask
        self.bit_supervision = bit_supervision
        if bit_supervision:
            #self.sbs = [nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 32), nn.LeakyReLU(), nn.Linear(32, 1))]
            self.sbs = [gnn.GATv2Conv(128, 1, edge_dim=2, heads=16, concat=False)]
            self.sbs_opt = torch.optim.Adam(self.sbs[0].parameters())
            
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.relu(self.gcn(x, edge_index, edge_attr))
        x_out = self.relu(self.gcn_main[0](x_out, edge_index, edge_attr))

        for i in range(1, self.iterations):
            x_out = self.relu(self.gcn_main[i](x_out, edge_index, edge_attr))
            if print_data:
                print(self.final(x_out)[-3:])
        
        if self.bit_supervision:
            x_bit = torch.squeeze(self.sbs[0](x_out.detach(), edge_index, edge_attr))

        x_out = torch.squeeze(self.final(x_out, edge_index, edge_attr))
        
        if print_data:
            print(x_out[-3:])
        
        if self.use_mask:
            x_out = x_out * data.mask
        if self.bit_supervision:
            return x_out, x_bit * (data.mask)
        return x_out


class NeuroSAT(nn.Module):
    def __init__(self, iterations=24, hidden_dim=64):
        super(NeuroSAT, self).__init__()
        self.L_init = nn.Linear(2, hidden_dim)
        self.C_init = nn.Linear(2, hidden_dim)

        self.L_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.C_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.msg_passing_net = gnn.MessagePassing(aggr='add')

        self.L_update = nn.LSTM(hidden_dim*2, hidden_dim)
        #self.L_norm   = nn.LayerNorm(64)
        self.C_update = nn.LSTM(hidden_dim, hidden_dim)
        #self.C_norm   = nn.LayerNorm(64)

        self.L_vote = MLP(hidden_dim, hidden_dim, 1)

        self.iterations = iterations
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        mask = torch.unsqueeze(data.mask, dim=-1)

        l_out = torch.unsqueeze(self.L_init(x)*(1-mask), dim=0) # (N+M(0), 64)
        c_out = torch.unsqueeze(self.C_init(x)*mask, dim=0) # (N(0)+M, 64)
        out = l_out + c_out

        l_hidden = torch.zeros_like(l_out).float().to(l_out.device)
        l_cell = torch.zeros_like(l_out).float().to(l_out.device)
        c_hidden = torch.zeros_like(c_out).float().to(c_out.device)
        c_cell = torch.zeros_like(c_out).float().to(c_out.device)

        for i in range(1, self.iterations):
            l_msg = self.L_msg(out) # Message from literal to clause
            l_msg = self.msg_passing_net.propagate(edge_index_sl, x=l_msg)
            
            _, (c_hidden, c_cell) = self.C_update(l_msg, (c_hidden, c_cell))
            temp_out = out * (1-mask) + c_hidden*mask
            
            c_msg = self.C_msg(temp_out)
            c_msg = self.msg_passing_net.propagate(edge_index_sl, x=c_msg)

            c_msg = torch.cat((self.flip(out, data.batch, data.mask), c_msg), dim=2)
            _, (l_hidden, l_cell) = self.L_update(c_msg, (l_hidden, l_cell))

            out = l_hidden*(1-mask) + c_hidden*mask
        #print(l_out.shape)
        x_out = self.L_vote(torch.squeeze(out*(1-mask)))
        #print(x_out.shape)
        
        if print_data:
            print(x_out[-3:])
        
        return x_out * (1-mask)
    
    def flip(self, msg, batch, mask):
        batch = batch.detach().to('cpu').numpy()
        msg_new = None
        msg = torch.squeeze(msg)
        for i in range(max(batch)+1):
            mask_now = mask[batch==i]
            msg_now = msg[batch==i]
            n_vars = int((mask_now==0).sum()/2)
            msg_now_new = torch.cat((msg_now[n_vars:2*n_vars, :], msg_now[:n_vars, :], msg_now[2*n_vars:]), dim=0)
            if not msg_new is None:
                msg_new = torch.cat((msg_new, msg_now_new), dim=0)
            else:
                msg_new = msg_now_new
        return torch.unsqueeze(msg_new,dim=0)

class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)

    return x


class GNNSat_NESY(nn.Module):
    def __init__(self, iterations=1):
        super(GNNSat_NESY, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.TransformerConv(2, 2, edge_dim=2)
        self.final = nn.Linear(2, 1)
        # Layer normalization
        self.ln = gnn.BatchNorm(2)
        self.softmax = nn.Softmax(dim=0)
    
    def getNeighborMax(self, x, var, edge_index, edge_attr):
        x_out = None
        for i in torch.arange(0, torch.max(edge_index).int()+1):
            temp = torch.sum(x[edge_index[0][edge_index[1]==i]]*edge_attr[edge_index[1]==i], dim=1)
            if temp.shape[0] != 0:
                temp_max = torch.unsqueeze(torch.max(temp, dim=0)[0], dim=0)
                x_out = torch.cat((x_out, torch.unsqueeze(torch.cat((temp_max, 1-temp_max)), dim=0)))
            elif x_out is None:
                x_out = torch.unsqueeze(x[i], dim=0)
            else:
                x_out = torch.cat((x_out, torch.unsqueeze(x[i], dim=0)))
        return x_out
    
    def forward(self, data, print_data=False):
        x = data.x
        var_index = list(range(1,data.edge_attr.shape[0],2))
        clause_index = list(range(0,data.edge_attr.shape[0],2))
        edge_index = data.edge_index[:, var_index]
        edge_attr = data.edge_attr[var_index]
        edge_attr_clause = data.edge_attr[clause_index]

        x_out = self.gcn(x, edge_index, edge_attr)
        x_out = self.ln(x_out)
        x_out = self.softmax(x_out)
        x_out = self.getNeighborMax(x_out, data.vars, edge_index, edge_attr_clause)
        for _ in range(self.iterations-1):
            x_out = self.gcn(x_out, edge_index, edge_attr)
            x_out = self.softmax(x_out)
            x_out = self.getNeighborMax(x_out, data.vars, edge_index, edge_attr_clause)
            if print_data:
                print(self.final(x_out)[-3:])

        x_out = torch.squeeze(self.final(x_out))
        if print_data:
            print(x_out[-3:])
        
        x_out = x_out * data.mask
        return x_out