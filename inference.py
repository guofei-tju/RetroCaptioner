import os
import re
import pickle
import numpy as np
import torch
import copy
import dgl
import random
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils import smiles2graph
from scipy import sparse as sp
from models.model import RetroModel

import warnings

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def var(a):
    return a.clone().detach()
    # return torch.tensor(a, requires_grad=False)


def rvar(a, beam_size=10):
    if len(a.size()) == 3:
        return var(a.repeat(1, beam_size, 1))
    else:
        return var(a.repeat(1, beam_size))


class OneStepRetroModel:
    def __init__(self, args, vocab_itos_src, vocab_itos_tgt, model_path):
        self.device = args.device
        self.src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
        self.tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]
        self.src_itos = vocab_itos_src
        self.tgt_itos = vocab_itos_tgt
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

        model = RetroModel(args, encoder_num_layers=8,
                           decoder_num_layers=8,
                           d_model=256, heads=8, d_ff=2048, dropout=0.1,
                           vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
                           shared_vocab=False,
                           src_pad_idx=self.src_pad_idx, tgt_pad_idx=self.tgt_pad_idx, device=self.device)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if 'module' in list(checkpoint['model'].keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model'])
        self.model = model.to(self.device)
        self.model.eval()

    def init_positional_encoding(self, g, pos_enc_dim, type_init):
        """
            Initializing positional encoding with RWPE
        """

        n = g.number_of_nodes()

        if type_init == 'rand_walk':
            # Geometric diffusion features with Random Walk
            A = g.adj_external(scipy_fmt='csr')
            # A = g.adjacency_matrix(scipy_fmt='csr')
            Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
            RW = A * Dinv
            M = RW

            # Iterate
            nb_pos_enc = pos_enc_dim
            PE = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(nb_pos_enc - 1):
                M_power = M_power * M
                PE.append(torch.from_numpy(M_power.diagonal()).float())
            PE = torch.stack(PE, dim=-1)
            g.ndata['pos_enc'] = PE

        return g

    def reorder_state_cache(slef, state_cache, selected_indices):
        """Reorder state_cache of the decoder
        params state_cache: list of indices
        params selected_indices: size (batch_size x beam_size)
        """
        batch_size, beam_size = len(selected_indices), len(selected_indices[0])
        indices_mapping = torch.arange(batch_size * beam_size,
                                       device=selected_indices[0].device).reshape(beam_size, batch_size).transpose(0, 1)
        reorder_indices = []
        for batch_i, indices in enumerate(selected_indices):
            reorder_indices.append(indices_mapping[batch_i, indices])
        reorder_indices = torch.stack(reorder_indices, dim=1).view(-1)

        new_state_cache = []
        for key in state_cache:
            if isinstance(state_cache[key], dict):
                for subkey in state_cache[key]:
                    state_cache[key][subkey] = state_cache[key][subkey][reorder_indices]

            elif isinstance(state_cache[key], torch.Tensor):
                state_cache[key] = state_cache[key][reorder_indices]
            else:
                raise Exception

    def smi_tokenizer(self, smi):
        """Tokenize a SMILES sequence or reaction"""
        pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        if smi != ''.join(tokens):
            print('ERROR:', smi, ''.join(tokens))
        assert smi == ''.join(tokens)
        return tokens

    def get_3d_adj(self, smi):
        mol = Chem.MolFromSmiles(smi)
        m2 = Chem.AddHs(mol)
        is_success = AllChem.EmbedMolecule(m2, enforceChirality=False)
        if is_success == -1:
            dist_adj = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))
        else:
            AllChem.MMFFOptimizeMolecule(m2)
            m2 = Chem.RemoveHs(m2)
            dist_adj = (-1 * AllChem.Get3DDistanceMatrix(m2))
        return dist_adj

    def preprocess(self, smiles_list):
        src_tokens = []
        graph_list = []
        graph_edge_list = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            smiles = Chem.MolToSmiles(mol)
            # smiles = smi
            src_token = self.smi_tokenizer(smiles)
            src_token = ['<UNK>'] + src_token
            src_tokens.append(torch.tensor([self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]))

            graph = smiles2graph(smiles)
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])
            dgl_graph.edata['feat'] = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            dgl_graph.ndata['feat'] = torch.from_numpy(graph['node_feat']).to(torch.int64)
            dgl_graph = self.init_positional_encoding(dgl_graph, 20, 'rand_walk')
            pro_adj = self.get_3d_adj(smiles)
            edges_dist = []
            for e in range(graph['edge_index'].shape[1]):
                edges_dist.append(pro_adj[graph['edge_index'][0][e], graph['edge_index'][1][e]])
            graph_list.append(dgl_graph)
            graph_edge_list.append(edges_dist)
        return src_tokens, graph_list, graph_edge_list

    def run(self, smi, topk=10, max_length=300):
        if type(smi) == str:
            smi = [smi]
        src_tokens, graph_list, graph_edge_list = self.preprocess(smi)
        pg_num = [i.number_of_nodes() for i in graph_list]
        pg_edge_num = [i.number_of_edges() for i in graph_list]
        dgl_graph = dgl.batch(graph_list).to(self.device)
        batch_pro_adj = torch.tensor(np.hstack(graph_edge_list), dtype=torch.float).view(-1, 1).to(self.device)

        max_len = max([len(i) for i in src_tokens])
        src_tensor = torch.ones((len(src_tokens), max_len), dtype=torch.long) * self.src_stoi['<pad>']
        for i in range(len(src_tokens)):
            src_tensor[i, :len(src_tokens[i])] = src_tokens[i]
        src_tensor = src_tensor.transpose(0, 1).to(self.device)

        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in pg_num]
        snorm_n = torch.cat(tab_snorm_n).sqrt().to(self.device)
        batch_size = len(src_tokens)
        pred_tokens = src_tensor.new_ones((batch_size, topk, max_length + 1), dtype=torch.long)
        pred_scores = src_tensor.new_zeros((batch_size, topk), dtype=torch.float)
        pred_tokens[:, :, 0] = 2
        batch2finish = {i: False for i in range(batch_size)}
        invalid_token_indices = [self.tgt_stoi['<unk>']]
        eos_idx = self.tgt_stoi['<eos>']
        # Encoder:
        with torch.no_grad():
            encoder_out, node_features, edge_feature, _ = self.model.cross_encoder(src_tensor,
                                                                                   (dgl_graph, snorm_n),
                                                                                   (pg_num, pg_edge_num),
                                                                                   adj_3d=batch_pro_adj)

            prior_encoder_out = encoder_out.transpose(0, 1).contiguous()

            src_repeat = rvar(src_tensor.data, beam_size=topk)
            memory_bank_repeat = rvar(prior_encoder_out.data, beam_size=topk)

            state_cache = {}
            for step in range(0, max_length):
                inp = pred_tokens.transpose(0, 1).contiguous().view(-1, pred_tokens.size(2))[:, :step + 1].transpose(0,
                                                                                                                     1)

                with torch.no_grad():
                    outputs, attn = self.model.decoder(src_repeat, inp, memory_bank_repeat, None,
                                                       state_cache=state_cache, step=step)
                    scores = self.model.generator(outputs[-1])

                unbottle_scores = scores.view(topk, batch_size, -1)

                # Avoid invalid token:
                unbottle_scores[:, :, invalid_token_indices] = -1e25

                # Avoid token that end earily
                if step < 2:
                    unbottle_scores[:, :, eos_idx] = -1e25

                # Beam Search:
                selected_indices = []
                for j in range(batch_size):
                    prev_score = pred_scores[j].clone()
                    batch_score = unbottle_scores[:, j]
                    num_words = batch_score.size(1)
                    # Get previous token to identify <eos>
                    prev_token = pred_tokens[j, :, step]
                    eos_index = prev_token.eq(eos_idx)
                    # Prevent <eos> sequence to have children
                    prev_score[eos_index] = -1e20

                    if topk == eos_index.sum():  # all beam has finished
                        pred_tokens[j, :, step + 1] = eos_idx
                        batch2finish[j] = True
                        selected_indices.append(torch.arange(topk, dtype=torch.long, device=src_tensor.device))
                    else:
                        beam_scores = batch_score + prev_score.unsqueeze(1).expand_as(batch_score)

                        if step == 0:
                            flat_beam_scores = beam_scores[0].view(-1)
                        else:
                            flat_beam_scores = beam_scores.view(-1)

                        # Select the top-k highest accumulative scores
                        k = topk - eos_index.sum().item()
                        best_scores, best_scores_id = flat_beam_scores.topk(k, 0, True, True)

                        # Freeze the tokens which has already finished
                        frozed_tokens = pred_tokens[j][eos_index]
                        if frozed_tokens.shape[0] > 0:
                            frozed_tokens[:, step + 1] = eos_idx
                        frozed_scores = pred_scores[j][eos_index]

                        # Update the rest of tokens
                        origin_tokens = pred_tokens[j][best_scores_id // num_words]
                        origin_tokens[:, step + 1] = best_scores_id % num_words

                        updated_scores = torch.cat([best_scores, frozed_scores])
                        updated_tokens = torch.cat([origin_tokens, frozed_tokens])

                        pred_tokens[j] = updated_tokens
                        pred_scores[j] = updated_scores

                        if eos_index.sum() > 0:
                            tmp_indices = src_tensor.new_zeros(topk, dtype=torch.long)
                            tmp_indices[:len(best_scores_id // num_words)] = (best_scores_id // num_words)
                            selected_indices.append(tmp_indices)
                        else:
                            selected_indices.append((best_scores_id // num_words))

                if selected_indices:
                    self.reorder_state_cache(state_cache, selected_indices)

                if sum(batch2finish.values()) == len(batch2finish):
                    break
        res = []
        for idx in range(batch_size):
            hypos = np.array(
                [''.join(s) for s in
                 [[self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]] for tokens in pred_tokens[idx]]])
            hypo_len = np.array([len(self.smi_tokenizer(ht)) for ht in hypos])
            new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
            ordering = np.argsort(new_pred_score)[::-1]
            pres = hypos[ordering]
            res.append([pres, new_pred_score[ordering]])

        res = res[0]
        reactants, score = [], []
        res[1] = np.exp(res[1])
        for r in range(len(res[0])):
            all_val = True
            for s in res[0][r].split('.'):
                if Chem.MolFromSmiles(s) is None:
                    all_val = False
                    break
            if all_val:
                reactants.append(res[0][r])
                score.append(res[1][r])
        total = sum(score)
        score = [s / total for s in score]
        return {'reactants': reactants,
                'scores': score,
                'template': ['Free'] * len(score)}
        return res


def get_paras():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intermediate_dir', type=str, default='intermediate', help='intermediate directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='a_model_550000.pt', help='checkpoint model file')

    parser.add_argument('--device', type=str, default='cuda:0', help='device GPU/CPU')
    parser.add_argument('--smiles', type=str, default='Brc1ccc(-c2ccc3ccccc3c2)cc1', help='input smiles')
    parser.add_argument('--beam_size', type=int, default=10, help='beam size')
    args = parser.parse_args()
    return args


def prepare_single_step_model(model_path):
    args = get_paras()
    args.proj_path = os.path.dirname(os.path.realpath(__file__))
    args.intermediate_dir = os.path.join(args.proj_path, args.intermediate_dir)
    with open(os.path.join(args.intermediate_dir, 'vocab.pk'), 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)

    args.d_model = 256
    # graph
    args.pos_enc_dim = 20
    args.pe_init = 'rand_walk'
    args.g_L = 4
    args.g_hidden_dim = 128
    args.g_residual = True
    args.g_edge_feat = True
    args.g_readout = 'mean'
    args.g_in_feat_dropout = 0.0
    args.g_dropout = 0.1
    args.g_batch_norm = True
    args.g_use_lapeig_loss = False
    args.g_alpha_loss = 1e-4
    args.g_lambda_loss = 1
    one_step = OneStepRetroModel(args, src_itos, tgt_itos, model_path)
    return one_step


if __name__ == '__main__':
    args = get_paras()
    setup_seed(0)
    smiles = args.smiles
    args.intermediate_dir = 'intermediate'
    args.checkpoint_dir = 'checkpoint'
    args.checkpoint = 'unknown_model.pt'
    args.proj_path = os.path.dirname(os.path.realpath(__file__))
    args.checkpoint_dir = os.path.join(args.proj_path, args.checkpoint_dir)
    args.intermediate_dir = os.path.join(args.proj_path, args.intermediate_dir)
    model_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    with open(os.path.join(args.intermediate_dir, 'vocab.pk'), 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)

    args.d_model = 256
    # graph
    args.pos_enc_dim = 20
    args.pe_init = 'rand_walk'
    args.g_L = 4
    args.g_hidden_dim = 128
    args.g_residual = True
    args.g_edge_feat = True
    args.g_readout = 'mean'
    args.g_in_feat_dropout = 0.0
    args.g_dropout = 0.1
    args.g_batch_norm = True
    args.g_use_lapeig_loss = False
    args.g_alpha_loss = 1e-4
    args.g_lambda_loss = 1

    one_step = OneStepRetroModel(args, src_itos, tgt_itos, model_path)


    print(f'input smiles: {args.smiles}')
    retacts = one_step.run([args.smiles])
    for res in retacts['reactants']:
        print(res)
