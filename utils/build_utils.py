from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from functools import partial
import os
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from dataset import RetroDataset
from models.model import RetroModel


def load_checkpoint(args, model):
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    optimizer = checkpoint['optim']
    step = checkpoint['step']
    step += 1
    return step, optimizer, model.to(args.device)


def build_model(args, vocab_itos_src, vocab_itos_tgt):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

    model = RetroModel(args,
                       encoder_num_layers=args.encoder_num_layers,
                       decoder_num_layers=args.decoder_num_layers,
                       d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
                       vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
                       shared_vocab=args.shared_vocab == 'True', gamma=args.gamma,
                       src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx, device=args.device)

    return model.to(args.device)


def build_iterator(args, train=True, sample=False, augment=False):
    if train:
        dataset = RetroDataset(args, mode='train', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class == 'True',
                               shared_vocab=args.shared_vocab == 'True', sample=sample, augment=augment)
        dataset_val = RetroDataset(args, mode='val', data_folder=args.data_dir,
                                   intermediate_folder=args.intermediate_dir,
                                   known_class=args.known_class == 'True',
                                   shared_vocab=args.shared_vocab == 'True', sample=sample)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample,  drop_last=True,
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return train_iter, val_iter, dataset.src_itos, dataset.tgt_itos

    else:
        dataset = RetroDataset(args, mode='test', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class == 'True',
                               shared_vocab=args.shared_vocab == 'True')
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device))
        return test_iter, dataset


def collate_fn(data, src_pad, tgt_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, alignment, reacts_graph, pg, gt_atom_idx, gt_edge_idx, reacts_adj, pro_adj = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    batched_graph = dgl.batch(pg).to(device)
    batched_reacts_graph = dgl.batch(reacts_graph).to(device)

    batch_pro_adj = torch.tensor(np.vstack(pro_adj), dtype=torch.float).view(-1, 1).to(device)
    batch_rea_adj = torch.tensor(np.vstack(reacts_adj), dtype=torch.float).view(-1, 1).to(device)

    rg_num = [i.number_of_nodes() for i in reacts_graph]
    pg_num = [i.number_of_nodes() for i in pg]
    rg_edge_num = [i.number_of_edges() for i in reacts_graph]
    pg_edge_num = [i.number_of_edges() for i in pg]

    anchor = torch.zeros([], device=device)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((len(data), max_tgt_length - 1, max_src_length), dtype=torch.float)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

    return new_src, new_tgt, new_alignment,batched_reacts_graph, batched_graph, (rg_num, rg_edge_num), (
               pg_num, pg_edge_num), gt_atom_idx, gt_edge_idx, batch_pro_adj, batch_rea_adj


def accumulate_batch(true_batch, src_pad=1, tgt_pad=1):
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]
    for batch in true_batch:
        src, tgt = batch[0], batch[1]
        src_max_length = max(src.shape[0], src_max_length)
        tgt_max_length = max(tgt.shape[0], tgt_max_length)
        entry_count += tgt.shape[1]

    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()

    new_context_alignment = torch.zeros((entry_count, tgt_max_length - 1, src_max_length)).float()

    # Graph packs:
    new_batch_graph, new_batch_graph_nums, new_reacts_graph, new_reacts_graph_nums = [], [], [], []
    new_batch_rc_atoms, new_batch_rc_edges = [], []
    new_reacts_edges, new_graph_edges = [], []
    new_batch_pro_adj,new_batch_rea_adj = [], []
    for i in range(len(true_batch)):
        src, tgt, context_alignment, reacts_graph, product_graph, \
        reacts_graph_num, product_graph_num, gt_atom_idx, gt_edge_idx, batch_pro_adj, batch_rea_adj = true_batch[i]
        new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
        new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
        new_context_alignment[batch_size * i: batch_size * (i + 1), :context_alignment.shape[1],
        :context_alignment.shape[2]] = context_alignment
        new_batch_graph.append(product_graph)
        new_reacts_graph.append(reacts_graph)
        new_batch_graph_nums += product_graph_num[0]
        new_reacts_graph_nums += reacts_graph_num[0]
        new_graph_edges += product_graph_num[1]
        new_reacts_edges += reacts_graph_num[1]
        new_batch_rc_atoms += list(gt_atom_idx)
        new_batch_rc_edges += list(gt_edge_idx)
        new_batch_pro_adj.append(batch_pro_adj)
        new_batch_rea_adj.append(batch_rea_adj)

    new_batch_graph = dgl.batch(new_batch_graph)
    new_reacts_graph = dgl.batch(new_reacts_graph)
    new_batch_pro_adj = torch.vstack(new_batch_pro_adj)
    new_batch_rea_adj = torch.vstack(new_batch_rea_adj)
    tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in new_reacts_graph_nums]
    reacts_snorm_n = torch.cat(tab_snorm_n).sqrt()

    tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in new_batch_graph_nums]
    pro_snorm_n = torch.cat(tab_snorm_n).sqrt()

    return new_src, new_tgt, new_context_alignment, (new_reacts_graph, reacts_snorm_n), (new_batch_graph, pro_snorm_n), (new_reacts_graph_nums, new_reacts_edges), \
           (new_batch_graph_nums, new_graph_edges), new_batch_rc_atoms, new_batch_rc_edges, new_batch_pro_adj,new_batch_rea_adj
