import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool

import numpy as np

class NeuroMUSX(nn.Module):
    
    def __init__(self, iterations=10):
        super(NeuroMUSX, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.GATv2Conv(2, 4, edge_dim=2, heads=16)
        gcn_main = []
        for _ in range(iterations):
            gcn_main.append(gnn.GATv2Conv(64, 4, edge_dim=2, heads=16))
        self.gcn_main = nn.Sequential(*gcn_main)
        self.final = gnn.GATv2Conv(64, 1, edge_dim=2, heads=16, concat=False)
        #self.final = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        # Layer normalization
        self.bn = gnn.BatchNorm(64)
        self.relu = nn.LeakyReLU()
        self.sbs = gnn.GATv2Conv(64, 1, edge_dim=2, heads=16, concat=False)
            
    
    def forward(self, data, batch, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.relu(self.bn(self.gcn(x, edge_index, edge_attr)))
        x_out = self.relu(self.gcn_main[0](x_out, edge_index, edge_attr))

        for i in range(1, self.iterations):
            x_out = self.relu(self.gcn_main[i](x_out, edge_index, edge_attr))
        
        x_out_mus = torch.squeeze(self.sbs(x_out, edge_index, edge_attr))
        x_out_sat = global_mean_pool(self.final(x_out, edge_index, edge_attr)[data.mask==0], data.batch[data.mask==0])
        x_out_sat = torch.squeeze(x_out_sat)

        return x_out_mus, x_out_sat

class NeuroMUSX_E(nn.Module):
    def __init__(self, iterations=1, d_in=2, d_hidden=64, d_edge=2, heads=16, dropout=0, add_self_loops=True):
        super(NeuroMUSX_E, self).__init__()
        self.iterations = iterations

        d_hidden_true = int(d_hidden/heads)
        self.gat_init = gnn.Sequential('x, edge_index, edge_attr',
                                        [(gnn.GATv2Conv(d_in, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'), 
                                        gnn.BatchNorm(d_hidden), #Pairnorm does not work well.
                                        nn.LeakyReLU()])
        
        gat_hidden = []
        for _ in range(iterations):
            gat_hidden.append(gnn.Sequential('x, edge_index, edge_attr', 
                                            [(gnn.GATv2Conv(d_hidden, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'),
                                            gnn.BatchNorm(d_hidden), 
                                            nn.LeakyReLU()]))
        self.gat_hidden = nn.Sequential(*gat_hidden)
        self.gat_out_mus = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=heads, add_self_loops=add_self_loops, concat=False)
        self.gat_out_sat = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=heads, add_self_loops=add_self_loops, concat=False)
            
    
    def forward(self, data, batch, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.gat_init(x, edge_index, edge_attr)
        
        for i in range(self.iterations):
            x_out = self.gat_hidden[i](x_out, edge_index, edge_attr)
        
        x_out_mus = torch.squeeze(self.gat_out_mus(x_out, edge_index, edge_attr))
        #x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr), data.batch)
        x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr)[data.mask==0], data.batch[data.mask==0])
        x_out_sat = torch.squeeze(x_out_sat)
        
        return x_out_mus, x_out_sat


class NeuroMUSX_V2(nn.Module):

    def __init__(self, iterations=10, d_in=2, d_hidden=64, d_edge=2, heads=16, dropout=0, add_self_loops=True, skip=2):
        super(NeuroMUSX_V2, self).__init__()
        if d_hidden % heads != 0:
            raise ValueError("d_hidden must be divisible by heads")
        
        self.iterations = iterations
        d_hidden_true = int(d_hidden/heads)
        self.gat_init = gnn.Sequential('x, edge_index, edge_attr',
                                        [(gnn.GATv2Conv(d_in, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'), 
                                        gnn.BatchNorm(d_hidden), #Pairnorm does not work well.
                                        nn.ELU()])
        
        gat_hidden = []
        for _ in range(iterations):
            gat_hidden.append(gnn.Sequential('x, edge_index, edge_attr', 
                                            [(gnn.GATv2Conv(d_hidden, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'),
                                            gnn.BatchNorm(d_hidden), 
                                            nn.ELU()]))
        self.gat_hidden = nn.Sequential(*gat_hidden)
        self.gat_out_mus = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=1, add_self_loops=add_self_loops)
        self.gat_out_sat = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=1, add_self_loops=add_self_loops)
        self.skip = skip

    def forward(self, data, batch, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.gat_init(x, edge_index, edge_attr)
        x_old = x_out
        
        for i in range(self.iterations):
            x_out = self.gat_hidden[i](x_out, edge_index, edge_attr)
            if (i+1) % self.skip == 0:
                x_out += x_old
        
        x_out_mus = torch.squeeze(self.gat_out_mus(x_out, edge_index, edge_attr))
        #x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr), data.batch)
        x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr)[data.mask==0], data.batch[data.mask==0])
        x_out_sat = torch.squeeze(x_out_sat)

        if False and print_data:
            print(x_out_sat)
            print(x_mus_mask)
            print(x_out_mus)
            exit()
        
        return x_out_mus, x_out_sat


class NeuroMUSX_V3(nn.Module):

    def __init__(self, iterations=10, d_in=2, d_hidden=64, d_edge=2, heads=16, dropout=0, add_self_loops=True, skip=2):
        super(NeuroMUSX_V3, self).__init__()
        if d_hidden % heads != 0:
            raise ValueError("d_hidden must be divisible by heads")
        
        self.iterations = iterations
        d_hidden_true = d_hidden
        self.gat_init = gnn.Sequential('x, edge_index, edge_attr',
                                        [(gnn.GINEConv(nn.Linear(d_in, d_hidden_true), edge_dim=d_edge, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'), 
                                        gnn.PairNorm(d_hidden), 
                                        nn.ELU()])
        
        gat_hidden = []
        for _ in range(iterations):
            gat_hidden.append(gnn.Sequential('x, edge_index, edge_attr', 
                                            [(gnn.GINEConv(nn.Linear(d_hidden, d_hidden_true), edge_dim=d_edge, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'),
                                            gnn.PairNorm(d_hidden), 
                                            nn.ELU()]))
        self.gat_hidden = nn.Sequential(*gat_hidden)
        self.gat_out_mus = gnn.GINEConv(nn.Linear(d_hidden, 1), edge_dim=d_edge, add_self_loops=add_self_loops)
        self.gat_out_sat = gnn.GINEConv(nn.Linear(d_hidden, heads), edge_dim=d_edge, add_self_loops=add_self_loops)
        self.sat_lin = nn.Linear(heads, 1)
        self.skip = skip

    def forward(self, data, batch, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.gat_init(x, edge_index, edge_attr)
        x_old = x_out
        
        for i in range(self.iterations):
            x_out = self.gat_hidden[i](x_out, edge_index, edge_attr)
            if (i+1) % self.skip == 0:
                x_out += x_old
        
        x_out_mus = torch.squeeze(self.gat_out_mus(x_out, edge_index, edge_attr))
        x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr), data.batch)
        x_out_sat = torch.squeeze(self.sat_lin(x_out_sat))
        if print_data and False:
            print(x_out_sat)
            exit()
        return x_out_mus, x_out_sat

class NeuroMUSX_V4(nn.Module):

    def __init__(self, iterations=10, d_in=2, d_hidden=128, d_edge=2, heads=16, dropout=0, add_self_loops=True, skip=2):
        super(NeuroMUSX_V4, self).__init__()
        if d_hidden % heads != 0:
            raise ValueError("d_hidden must be divisible by heads")
        
        self.iterations = iterations
        d_hidden_true = int(d_hidden/heads)
        self.gat_init = gnn.Sequential('x, edge_index, edge_attr',
                                        [(gnn.GATv2Conv(d_in, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'), 
                                        #gnn.LayerNorm(d_hidden), 
                                        gnn.BatchNorm(d_hidden), 
                                        nn.ELU()])
        
        gat_hidden = []
        for _ in range(iterations):
            gat_hidden.append(gnn.Sequential('x, edge_index, edge_attr', 
                                            [(gnn.GATv2Conv(d_hidden, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'),
                                            #gnn.LayerNorm(d_hidden), 
                                            gnn.BatchNorm(d_hidden), 
                                            nn.ELU()]))
        self.gat_hidden = nn.Sequential(*gat_hidden)
        self.gat_out_mus = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=1, add_self_loops=add_self_loops)
        self.gat_out_sat = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=heads, add_self_loops=add_self_loops)
        self.sat_lin = nn.Linear(heads, 1)
        self.skip = skip

    def forward(self, data, batch, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.gat_init(x, edge_index, edge_attr)
        x_old = x_out
        
        for i in range(self.iterations):
            x_out = self.gat_hidden[i](x_out, edge_index, edge_attr)
            if (i+1) % self.skip == 0:
                x_out += x_old
        
        x_out_mus = torch.squeeze(self.gat_out_mus(x_out, edge_index, edge_attr))
        #x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr)[data.mask==0], data.batch[data.mask==0])
        #x_out_sat = torch.squeeze(self.sat_lin(x_out_sat))
        
        x_out_sat_pre = self.gat_out_sat(x_out, edge_index, edge_attr)
        x_out_sat = torch.squeeze(self.sat_lin(torch.mean(x_out_sat_pre[batch==0], dim=0, keepdim=True)), dim=-1)#/torch.sum(1-data.mask[batch==0], dim=0, keepdim=True)
        x_mus_mask = (1-torch.sigmoid(x_out_sat)) * torch.tensor(np.ones_like(batch[batch==0])).to(x_out_sat.device)
        #print(torch.tensor(np.ones_like(batch[batch==0])).to(x_out_sat.device))
        # 1: Sat, 0: Unsat
        for i in range(1, np.max(batch) + 1):
            x_temp = torch.squeeze(self.sat_lin(torch.mean(x_out_sat_pre[batch==i], dim=0, keepdim=True)), dim=-1)#/torch.sum(1-data.mask[batch==0], dim=0, keepdim=True)
            x_out_sat = torch.cat((x_out_sat, x_temp))
            x_mus_mask = torch.cat((x_mus_mask, (1-torch.sigmoid(x_temp))*torch.tensor(np.ones_like(batch[batch==i])).to(x_out_sat.device)))
        # Unsat: 1, Sat: 0
        x_out_mus = x_out_mus + (x_mus_mask - 0.5)
        if False and print_data:
            print(x_out_sat)
            print(x_mus_mask)
            print(x_out_mus)
            exit()
        
        return x_out_mus, x_out_sat



class NeuroSAT(nn.Module):
    def __init__(self, iterations=24, hidden_dim=64):
        super(NeuroSAT, self).__init__()
        self.L_init = nn.Linear(2, hidden_dim)
        self.C_init = nn.Linear(2, hidden_dim)

        self.L_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.C_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.msg_passing_net = gnn.MessagePassing(aggr='add')

        self.L_update = nn.LSTM(hidden_dim*2, hidden_dim)
        self.C_update = nn.LSTM(hidden_dim, hidden_dim)
        self.L_norm   = nn.LayerNorm(hidden_dim)
        self.C_norm   = nn.LayerNorm(hidden_dim)

        self.L_vote = MLP(hidden_dim, hidden_dim, 1)

        self.iterations = iterations
    
    def forward(self, data, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        mask = torch.unsqueeze(data.mask, dim=-1).float()

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
            c_hidden = self.C_norm(c_hidden)
            temp_out = out * (1-mask) + c_hidden*mask
            
            c_msg = self.C_msg(temp_out)
            c_msg = self.msg_passing_net.propagate(edge_index_sl, x=c_msg)
            c_msg = torch.cat((self.flip(out, data.batch, data.mask), c_msg), dim=2)
            _, (l_hidden, l_cell) = self.L_update(c_msg, (l_hidden, l_cell))
            l_hidden = self.L_norm(l_hidden)

            out = l_hidden*(1-mask) + c_hidden*mask
        #print(l_out.shape)
        x_out = self.L_vote(torch.squeeze(out*(1-mask)))
        #print(x_out.shape)
        
        if print_data:
            print(x_out[-3:])
        return torch.squeeze(global_mean_pool(x_out[data.mask==0], data.batch[data.mask==0]))
    
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


class NeuroMUSX_NeSy(nn.Module):


    def __init__(self, iterations=10, d_in=2, d_hidden=64, d_edge=2, heads=16, dropout=0, add_self_loops=True, skip=2):
        super(NeuroMUSX_V2, self).__init__()
        if d_hidden % heads != 0:
            raise ValueError("d_hidden must be divisible by heads")
        
        self.iterations = iterations
        d_hidden_true = int(d_hidden/heads)
        self.gat_init = gnn.Sequential('x, edge_index, edge_attr',
                                        [(gnn.GATv2Conv(d_in, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'), 
                                        gnn.BatchNorm(d_hidden),
                                        nn.ELU()])
        
        gat_hidden = []
        for _ in range(iterations):
            gat_hidden.append(gnn.Sequential('x, edge_index, edge_attr', 
                                            [(gnn.GATv2Conv(d_hidden, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout, add_self_loops=add_self_loops), 'x, edge_index, edge_attr -> x'),
                                            gnn.BatchNorm(d_hidden), 
                                            nn.ELU()]))
        self.gat_hidden = nn.Sequential(*gat_hidden)
        self.gat_out_mus = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=1, add_self_loops=add_self_loops)
        self.gat_out_sat = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=1, add_self_loops=add_self_loops)
        self.skip = skip

    def forward(self, data, batch, print_data=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.gat_init(x, edge_index, edge_attr)
        x_old = x_out
        
        for i in range(self.iterations):
            x_out = self.gat_hidden[i](x_out, edge_index, edge_attr)
            if (i+1) % self.skip == 0:
                x_out += x_old
        
        x_out_mus = torch.squeeze(self.gat_out_mus(x_out, edge_index, edge_attr))
        #x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr), data.batch)
        x_out_sat = global_mean_pool(self.gat_out_sat(x_out, edge_index, edge_attr)[data.mask==0], data.batch[data.mask==0])
        x_out_sat = torch.squeeze(x_out_sat)

        if False and print_data:
            print(x_out_sat)
            print(x_mus_mask)
            print(x_out_mus)
            exit()
        
        return x_out_mus, x_out_sat