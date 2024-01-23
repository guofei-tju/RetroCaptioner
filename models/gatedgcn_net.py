import dgl
import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from scipy import sparse as sp
from scipy.sparse.linalg import norm

"""
    GatedGCN and GatedGCN-LSPE
    
"""
from .gatedgcn_layer import GatedGCNLayer
from .gatedgcn_lspe_layer import GatedGCNLSPELayer

PI = 3.14159
A = (2 * PI) ** 0.5


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
class GatedGCNNet(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        hidden_dim = args.g_hidden_dim
        out_dim = args.d_model
        in_feat_dropout = args.g_in_feat_dropout
        dropout = args.g_dropout
        self.n_layers = args.g_L
        self.readout = args.g_readout
        self.batch_norm = args.g_batch_norm
        self.residual = args.g_residual
        self.edge_feat = args.g_edge_feat

        self.pe_init = args.pe_init

        self.use_lapeig_loss = args.g_use_lapeig_loss
        self.lambda_loss = args.g_lambda_loss
        self.alpha_loss = args.g_alpha_loss

        self.pos_enc_dim = args.pos_enc_dim
        self.device = device

        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        # self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        self.embedding_h = AtomEncoder(hidden_dim)
        self.embedding_e = BondEncoder(emb_dim=hidden_dim)

        self.use_3d_info = True
        if self.use_3d_info:
            self.spatial3d_encoder = GaussianBondLayer(hidden_dim, means=(0, 3), stds=(0.1, 10))

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout,
                                                           self.batch_norm, residual=self.residual) for _ in
                                         range(self.n_layers - 1)])
            self.layers.append(GatedGCNLSPELayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual))
        else:
            # NoPE or LapPE
            self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, residual=self.residual, graph_norm=False) for _
                                         in range(self.n_layers - 1)])
            self.layers.append(
                GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual, graph_norm=False))

        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim + self.pos_enc_dim, out_dim)

        self.g = None  # For util; To be accessed in loss() function

    def embedd_nodes(self, g, h, p, e, snorm_n, adj_3d=None):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p)

        if self.pe_init == 'lap_pe':
            h = h + p
            p = None

        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)
        if adj_3d is not None:
            e = e + self.spatial3d_encoder(adj_3d)

        # convnets
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)

        g.ndata['h'] = h

        if self.pe_init == 'rand_walk':
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)
            # g.ndata['p'] = p
            # means = dgl.mean_nodes(g, 'p')
            # batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            # p = p - batch_wise_p_means
            #
            # # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            # g.ndata['p'] = p
            # g.ndata['p2'] = g.ndata['p'] ** 2
            # norms = dgl.sum_nodes(g, 'p2')
            # norms = torch.sqrt(norms)
            # batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0) + 1e-9
            # p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p

            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'], g.ndata['p']), dim=-1))
            g.ndata['h'] = hp

        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        self.g = g  # For util; To be accessed in loss() function

        return g.ndata['h'], hg, e

    def forward(self, g, h, p, e, snorm_n):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p)

        if self.pe_init == 'lap_pe':
            h = h + p
            p = None

        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)

        g.ndata['h'] = h

        if self.pe_init == 'rand_walk':
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p'] ** 2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p

            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'], g.ndata['p']), dim=-1))
            g.ndata['h'] = hp

        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        self.g = g  # For util; To be accessed in loss() function

        return self.MLP_layer(hg), g

    def loss(self, scores, targets):

        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)

        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian 
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro') ** 2).float().to(self.device)

            loss_b = (loss_b_1 + self.lambda_loss * loss_b_2) / (self.pos_enc_dim * batch_size * n)

            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b
        else:
            loss = loss_a

        return loss

def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (A * std)

class GaussianBondLayer(nn.Module):
    def __init__(self, nhead=16, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.nhead = nhead
        self.means = nn.Embedding(1, nhead)
        self.stds = nn.Embedding(1, nhead)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x + bias
        x = x.expand(-1, self.nhead)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

