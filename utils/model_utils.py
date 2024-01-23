import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def reallocate_batch(batch, location='cpu'):
    batch = list(batch)
    for i in range(len(batch)):
        batch[i] = batch[i].to(location)
    return tuple(batch)


def validate(args,model, val_iter, pad_idx=1):
    pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
    pred_arc_list, gt_arc_list = [], []
    pred_brc_list, gt_brc_list = [], []
    model.eval()
    for batch in tqdm(val_iter):
        src, tgt, gt_context_alignment, reacts_graph, product_graph, \
        reacts_graph_num, product_graph_num, gt_atom_idx, gt_edge_idx, new_batch_adj, rea_batch_adj = batch

        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in product_graph_num[0]]
        reacts_snorm_n = torch.cat(tab_snorm_n).sqrt().to(args.device)
        # Infer:
        with torch.no_grad():
            scores, atom_rc_scores, bond_rc_scores, context_alignment, center_loss = \
                model(src, tgt, (product_graph,reacts_snorm_n), product_graph_num,new_batch_adj, gt_atom_idx)

        # Atom-level reaction center accuracy:
        gt_atom_rc_label = np.array([])
        for j in range(len(product_graph_num[0])):
            l = np.array([False] * product_graph_num[0][j])
            l[gt_atom_idx[j]] = True
            gt_atom_rc_label = np.hstack((gt_atom_rc_label, l))
        pred_arc = (atom_rc_scores.squeeze(1) > 0.5).bool()
        pred_arc_list += list(pred_arc.view(-1).cpu().numpy())
        gt_arc_list += list(gt_atom_rc_label)

        # Bond-level reaction center accuracy:
        gt_edge_rc_label = np.array([])
        for j in range(len(product_graph_num[1])):
            l = np.array([False] * product_graph_num[1][j])
            l[gt_edge_idx[j]] = True
            gt_edge_rc_label = np.hstack((gt_edge_rc_label, l))
        pred_brc = (bond_rc_scores > 0.5).bool()
        pred_brc_list += list(pred_brc.view(-1).cpu().numpy())
        gt_brc_list += list(gt_edge_rc_label)

        # Token accuracy:
        pred_token_logit = scores.view(-1, scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)

    if bond_rc_scores is not None:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               np.mean(np.array(pred_brc_list) == np.array(gt_brc_list)), \
               (pred_tokens == gt_tokens).float().mean().item()
    else:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               0, \
               (pred_tokens == gt_tokens).float().mean().item()
