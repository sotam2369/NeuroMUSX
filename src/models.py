import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops

import numpy as np

class NeuroMUSX(nn.Module):
    def __init__(self, iterations=1):
        super(NeuroMUSX, self).__init__()
        self.iterations = iterations
        self.gcn = gnn.GATv2Conv(2, 8, edge_dim=2, heads=16)
        gcn_main = []
        for _ in range(iterations):
            gcn_main.append(gnn.GATv2Conv(128, 8, edge_dim=2, heads=16))
        self.gcn_main = nn.Sequential(*gcn_main)
        self.final = gnn.GATv2Conv(128, 1, edge_dim=2, heads=16, concat=False)
        self.relu = nn.LeakyReLU()
        self.sbs = [gnn.GATv2Conv(128, 1, edge_dim=2, heads=16, concat=False)]
            
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch.detach().cpu().numpy()

        x_out = self.relu(self.gcn(x, edge_index, edge_attr))
        x_out = self.relu(self.gcn_main[0](x_out, edge_index, edge_attr))

        for i in range(1, self.iterations):
            x_out = self.relu(self.gcn_main[i](x_out, edge_index, edge_attr))
        
        x_bit = torch.squeeze(self.sbs[0](x_out.detach(), edge_index, edge_attr))
        x_out = torch.squeeze(self.final(x_out, edge_index, edge_attr))
        
        
        x_out = x_out * data.mask
        x_out_sat = torch.mean(x_bit[batch==0], dim=0, keepdim=True)
        for i in range(1, np.max(batch) + 1):
            x_out_sat = torch.cat((x_out_sat, torch.mean(x_bit[batch==i], dim=0, keepdim=True)))
        return x_out, x_out_sat


class NeuroMUSX_V2(nn.Module):

    def __init__(self, iterations=10, d_in=2, d_hidden=128, d_edge=2, heads=8, dropout=0):
        super(NeuroMUSX_V2, self).__init__()
        if d_hidden % heads != 0:
            raise ValueError("d_hidden must be divisible by heads")
        
        self.iterations = iterations
        d_hidden_true = int(d_hidden/heads)
        self.gat_init = gnn.Sequential('x, edge_index, edge_attr',
                                        [(gnn.GATv2Conv(d_in, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout), 'x, edge_index, edge_attr -> x'), 
                                        gnn.BatchNorm(d_hidden), 
                                        nn.LeakyReLU()])
        
        gat_hidden = []
        for _ in range(iterations):
            gat_hidden.append(gnn.Sequential('x, edge_index, edge_attr', 
                                            [(gnn.GATv2Conv(d_hidden, d_hidden_true, edge_dim=d_edge, heads=heads, dropout=dropout), 'x, edge_index, edge_attr -> x'),
                                            gnn.BatchNorm(d_hidden), 
                                            nn.LeakyReLU()]))
        self.gat_hidden = nn.Sequential(*gat_hidden)
        self.gat_out_mus = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=heads, dropout=dropout, concat=False)
        self.gat_out_sat = gnn.GATv2Conv(d_hidden, 1, edge_dim=d_edge, heads=heads, dropout=dropout, concat=False)

    def forward(self, data, batch):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_out = self.gat_init(x, edge_index, edge_attr)
        
        for i in range(self.iterations):
            x_out = self.gat_hidden[i](x_out, edge_index, edge_attr)
        
        x_out_mus = torch.squeeze(self.gat_out_mus(x_out, edge_index, edge_attr))
        x_out_sat_pre = torch.squeeze(self.gat_out_sat(x_out, edge_index, edge_attr))
        
        x_out_sat = torch.mean(x_out_sat_pre[batch==0], dim=0, keepdim=True)
        for i in range(1, np.max(batch) + 1):
            x_out_sat = torch.cat((x_out_sat, torch.mean(x_out_sat_pre[batch==i], dim=0, keepdim=True)))
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