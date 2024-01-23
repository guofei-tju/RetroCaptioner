import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm

import dgl
from ogb.utils import smiles2graph
from rdkit import Chem

import torch
from torch.utils.data import Dataset
from utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph, canonical_smiles_with_am, \
    remove_am_without_canonical, extract_relative_mapping, get_nonreactive_mask, randomize_smiles_with_am, get_3d_adj
from utils.graph_utils import init_positional_encoding

from joblib import Parallel, delayed


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))


class RetroDataset(Dataset):
    def __init__(self, args, mode, data_folder='./data', intermediate_folder='./intermediate',
                 known_class=False, shared_vocab=False, augment=False, sample=False):
        self.data_folder = data_folder
        self.args = args

        assert mode in ['train', 'test', 'val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, data_folder))
        vocab_file = ''
        if 'full' in self.data_folder:
            vocab_file = 'full_'
        if shared_vocab:
            vocab_file += 'vocab_share.pk'
        else:
            vocab_file += 'vocab.pk'

        if mode != 'train':
            assert vocab_file in os.listdir(intermediate_folder)
            with open(os.path.join(intermediate_folder, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)
            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
            self.data = pd.read_csv(os.path.join(data_folder, '{}.csv'.format(mode)))
            if sample:
                self.data = self.data.sample(n=200, random_state=0)
                self.data.reset_index(inplace=True, drop=True)
        else:
            train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'))
            val_data = pd.read_csv(os.path.join(data_folder, 'val.csv'))
            if sample:
                train_data = train_data.sample(n=1000, random_state=0)
                train_data.reset_index(inplace=True, drop=True)
                val_data = val_data.sample(n=200, random_state=0)
                val_data.reset_index(inplace=True, drop=True)
            if vocab_file not in os.listdir(intermediate_folder):
                print('Building vocab...')
                raw_data = pd.concat([val_data, train_data])
                raw_data.reset_index(inplace=True, drop=True)
                prods, reacts = self.build_vocab_from_raw_data(raw_data)
                if self.shared_vocab:  # Shared src and tgt vocab
                    itos = set()
                    for i in range(len(prods)):
                        itos.update(smi_tokenizer(prods[i]))
                        itos.update(smi_tokenizer(reacts[i]))
                    itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    itos.add('<UNK>')
                    itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(list(itos))
                    self.src_itos, self.tgt_itos = itos, itos
                else:  # Non-shared src and tgt vocab
                    self.src_itos, self.tgt_itos = set(), set()
                    for i in range(len(prods)):
                        self.src_itos.update(smi_tokenizer(prods[i]))
                        self.tgt_itos.update(smi_tokenizer(reacts[i]))
                    self.src_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    self.src_itos.add('<UNK>')
                    self.src_itos = ['<unk>', '<pad>'] + sorted(
                        list(self.src_itos))
                    self.tgt_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(
                        list(self.tgt_itos))
                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

                with open(os.path.join(intermediate_folder, vocab_file), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(intermediate_folder, vocab_file), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

            self.data = eval('{}_data'.format(mode))

        # Build and load processed data into lmdb
        if 'cooked_{}.lmdb'.format(self.mode) not in os.listdir(intermediate_folder):
            self.build_processed_data(self.data, intermediate_folder)
        self.env = lmdb.open(os.path.join(intermediate_folder, 'cooked_{}.lmdb'.format(self.mode)),
                             max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279))  # pre-computed

    def build_vocab_from_raw_data(self, raw_data):
        reactions = raw_data['rxn_smiles'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, p = rxn.split('>>')
            if not r or not p:
                continue

            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True)
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)
        return prods, reacts

    def build_processed_data(self, raw_data, path):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['rxn_smiles'].to_list()

        env = lmdb.open(os.path.join(path, 'cooked_{}.lmdb'.format(self.mode)),
                        map_size=1099511627776)

        parallel = Parallel(n_jobs=-1)
        ps, rs, rts = [], [], []
        for i in (range(len(reactions))):
            ps.append(reactions[i].split('>>')[1])
            rs.append(reactions[i].split('>>')[0])
            rts.append('<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>')
        all_results = parallel(delayed(self.parse_smi_wrapper)((ps[i], rs[i], rts[i])) for i in (range(len(reactions))))
        print(len(all_results))

        with env.begin(write=True) as txn:
            for i in range(len(all_results)):
                p, r, rt = ps[i], rs[i], rts[i]

                result = all_results[i]
                if result is not None:
                    src, tgt, context_align, graphs = result

                    p_key = '{} {}'.format(i, clear_map_number(p))
                    processed = {
                        'src': src,
                        'tgt': tgt,
                        'context_align': context_align,
                        'reacts_graph': graphs[0],
                        'product_graph': graphs[1],
                        'product_gt_changed_idx': graphs[2],
                        'product_gt_edge_index': graphs[3],
                        'reacts_edge_3d': graphs[4],
                        'product_edge_3d': graphs[5],
                        'raw_product': p,
                        'raw_reactants': r,
                        'reaction_class': rt
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p_key))
                        continue
                else:
                    print('Warning. Process Failed.')

        return

    def parse_smi_wrapper(self, task):
        prod, reacts, react_class = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, build_vocab=False, randomize=False):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Get the nonreactive masking based on atom-mapping
        gt_nonreactive_mask, gt_changed_idx = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1)
        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        if randomize:
            return src_token, tgt_token, gt_context_attn

        reacts_graph = smiles2graph(reacts)
        dgl_reacts_graph = dgl.graph((reacts_graph['edge_index'][0], reacts_graph['edge_index'][1]),
                                     num_nodes=reacts_graph['num_nodes'])
        dgl_reacts_graph.edata['feat'] = torch.from_numpy(reacts_graph['edge_feat']).to(torch.int64)
        dgl_reacts_graph.ndata['feat'] = torch.from_numpy(reacts_graph['node_feat']).to(torch.int64)
        reacts_adj = get_3d_adj(reacts)
        edges_dist = []
        for e in range(reacts_graph['edge_index'].shape[1]):
            edges_dist.append(reacts_adj[reacts_graph['edge_index'][0][e], reacts_graph['edge_index'][1][e]])
        batch_reacts_adj = np.array(edges_dist).reshape(-1, 1)

        graph = smiles2graph(cano_prod)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])

        dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])
        dgl_graph.edata['feat'] = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        dgl_graph.ndata['feat'] = torch.from_numpy(graph['node_feat']).to(torch.int64)
        pro_adj = get_3d_adj(cano_prod)
        gt_edge_index, edges_dist = [], []
        for e in range(graph['edge_index'].shape[1]):
            edges_dist.append(pro_adj[graph['edge_index'][0][e], graph['edge_index'][1][e]])
            if graph['edge_index'][0][e] in gt_changed_idx and graph['edge_index'][1][e] in gt_changed_idx:
                gt_edge_index.append(e)
        batch_pro_adj = np.array(edges_dist).reshape(-1, 1)

        if self.args.pe_init == 'rand_walk':
            dgl_graph = init_positional_encoding(dgl_graph, self.args.pos_enc_dim, self.args.pe_init)
            dgl_reacts_graph = init_positional_encoding(dgl_reacts_graph, self.args.pos_enc_dim, self.args.pe_init)

        p_r_graphs = [dgl_reacts_graph, dgl_graph, gt_changed_idx, gt_edge_index, batch_reacts_adj, batch_pro_adj]

        return src_token, tgt_token, gt_context_attn, p_r_graphs

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return len(self.product_keys)

    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        p_key = p_key.decode().split(' ')[1]
        reacts_graph = processed['reacts_graph']
        product_graph = processed['product_graph']
        product_gt_changed_idx = processed['product_gt_changed_idx']
        product_gt_edge_index = processed['product_gt_edge_index']
        reacts_adj = processed['reacts_edge_3d']
        pro_adj = processed['product_edge_3d']

        src, tgt, context_alignment = processed['src'], processed['tgt'], processed['context_align']

        p = np.random.rand()
        if self.mode == 'train' and p > 0.5 and self.augment:
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            try:
                src, tgt, context_alignment = self.parse_smi(prod, react, rt, randomize=True)
            except:
                src, tgt, context_alignment = processed['src'], processed['tgt'], processed['context_align']

        # Make sure the reaction class is known/unknown
        if self.known_class:
            src[0] = self.src_stoi[processed['reaction_class']]
        else:
            src[0] = self.src_stoi['<UNK>']

        return src, tgt, context_alignment, reacts_graph, product_graph, product_gt_changed_idx, product_gt_edge_index, reacts_adj, pro_adj
