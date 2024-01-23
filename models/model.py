from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

from .decoder import TransformerDecoder
from .embedding import Embedding
from .encoder import TransformerEncoder
from .module import MultiHeadedAttention
from .gatedgcn_net import GatedGCNNet


class RetroModel(nn.Module):
    def __init__(self, args, encoder_num_layers, decoder_num_layers, d_model, heads, d_ff, dropout,
                 vocab_size_src, vocab_size_tgt, shared_vocab, gamma=2, src_pad_idx=1, tgt_pad_idx=1, device='cpu'):
        super(RetroModel, self).__init__()
        self.args = args
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.gamma = gamma
        self.dropout = dropout
        self.shared_vocab = shared_vocab
        self.device = device
        if shared_vocab:
            assert vocab_size_src == vocab_size_tgt and src_pad_idx == tgt_pad_idx
            self.embedding_src = self.embedding_tgt = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model,
                                                                padding_idx=src_pad_idx)
        else:
            self.embedding_src = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx)
            self.embedding_tgt = Embedding(vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx)

        self.embedding_bond = nn.Linear(7, d_model)

        text_en_num_layers = encoder_num_layers // 2
        multihead_attn_modules_en_text = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(text_en_num_layers)])

        encoder_num_layers = encoder_num_layers - text_en_num_layers
        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(encoder_num_layers)])

        multihead_attn_modules_de = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(decoder_num_layers)])

        self.text_encoder = TransformerEncoder(num_layers=text_en_num_layers,
                                               d_model=d_model, heads=heads,
                                               d_ff=d_ff, dropout=dropout,
                                               embeddings=self.embedding_src,
                                               embeddings_bond=self.embedding_bond,
                                               attn_modules=multihead_attn_modules_en_text)

        self.encoder = TransformerEncoder(num_layers=encoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_src,
                                          embeddings_bond=self.embedding_bond,
                                          attn_modules=multihead_attn_modules_en)

        self.decoder = TransformerDecoder(num_layers=decoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_tgt,
                                          self_attn_modules=multihead_attn_modules_de)

        self.gnn_node = GatedGCNNet(args, self.device).to(self.device)

        self.centers = nn.Embedding(2, self.d_model).to(self.device)

        # define the embedding layer of retro model
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pooling_layer = AvgPooling()

        self.gnn_z = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model)
        )

        self.atom_rc_identifier = nn.Sequential(nn.Linear(d_model, 1),
                                                nn.Sigmoid())
        self.bond_rc_identifier = nn.Sequential(nn.Linear(d_model, 1),
                                                nn.Sigmoid())

        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                       nn.LogSoftmax(dim=-1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, product_graph=None, product_graph_num=None, adj_3d=None, rc_atom=None):
        encoder_out, node_features, edge_feature, batch_graph_embed_z = self.cross_encoder(src, product_graph,
                                                                                           product_graph_num,
                                                                                           adj_3d=adj_3d)
        atom_rc_scores = self.atom_rc_identifier(node_features)
        bond_rc_scores = self.bond_rc_identifier(edge_feature) if edge_feature is not None else None

        encoder_out = encoder_out.transpose(0, 1).contiguous()
        decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out)

        generative_scores = self.generator(decoder_out)

        if rc_atom:
            index_matrix = torch.zeros(batch_graph_embed_z.shape[:2]).long().to(self.device)
            for i in range(len(rc_atom)):
                sub_indices = rc_atom[i]
                sub_indices_tensor = torch.tensor(sub_indices)
                index_matrix[i, sub_indices_tensor] = 1
            centers = self.centers(index_matrix)
            dic = batch_graph_embed_z - centers
            for i in range(len(rc_atom)):
                dic[i, rc_atom[i], :] = 0
            distances = torch.norm(dic, dim=2).sum()
            c_distance = torch.norm(self.centers(torch.tensor(0).to(self.device)) - self.centers(
                torch.tensor(1).to(self.device)) + self.gamma)
            return generative_scores, atom_rc_scores, bond_rc_scores, top_aligns, distances + c_distance
        return generative_scores, atom_rc_scores, bond_rc_scores, top_aligns

    def cross_encoder(self, seq_index, graph, graph_num, adj_3d=None):
        batch_graph_embed, node_feature, edge_feature = self.graph_encoder(graph, graph_num, adj_3d)  # [B, N, d]
        text_out, _ = self.text_encoder(seq_index, bond=None)
        batch_graph_embed_z = self.gnn_z(batch_graph_embed)
        text_graph_sim = torch.matmul(text_out.transpose(0, 1), batch_graph_embed_z.transpose(1, 2))
        for i in range(len(graph_num[0])):
            text_graph_sim[i, :, graph_num[0][i]:] = -1e18

        text_graph_sim = self.softmax(text_graph_sim).sum(dim=1)
        for i in range(len(graph_num[0])):
            text_graph_sim[i, graph_num[0][i]:] = -1e18

        text_graph_sim = self.softmax(text_graph_sim).unsqueeze(dim=2)
        weighted_batch_graph_embed = text_graph_sim * batch_graph_embed
        encoder_out, _ = self.encoder(seq_index, bond=None, emb=text_out,
                                      graph_embed=weighted_batch_graph_embed)  # [L, B, d]
        encoder_out = encoder_out.transpose(0, 1)
        return encoder_out, node_feature, edge_feature, batch_graph_embed_z

    def graph_readout(self, batch_graphs, adj_3d=None):
        batch_graphs, norm = batch_graphs
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(self.device)
        except KeyError:
            batch_pos_enc = None
        if self.args.pe_init == 'lap_pe':
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(self.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        h_node, hg, e_node = self.gnn_node.embedd_nodes(batch_graphs, batch_graphs.ndata["feat"], batch_pos_enc,
                                                        batch_graphs.edata["feat"], norm.to(self.device), adj_3d)
        h_node = self.layer_norm(h_node)
        return h_node, hg, e_node

    def graph_encoder(self, batch_graphs, batch_graph_nums, adj_3d):
        h_node, _, e_node = self.graph_readout(batch_graphs, adj_3d)
        cur_graph_num = 0
        batch_graph_embed = torch.tensor([]).to(self.device)
        batch_graph_nums, batch_graph_edge_nums = batch_graph_nums
        length = max(batch_graph_nums)
        for i in range(len(batch_graph_nums)):
            cur_graph_h = h_node[cur_graph_num:cur_graph_num + batch_graph_nums[i], :]
            pad_embed = torch.zeros((length - cur_graph_h.shape[0], self.d_model)).to(self.device)
            cur_embed = torch.cat([cur_graph_h, pad_embed], dim=0).unsqueeze(0)

            cur_graph_num += batch_graph_nums[i]
            batch_graph_embed = torch.cat([batch_graph_embed, cur_embed], dim=0)

        return batch_graph_embed, h_node, e_node
