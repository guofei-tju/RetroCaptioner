import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
sys.path.append(os.path.dirname(path))

import pickle
import numpy as np
from datetime import datetime

from utils.build_utils import build_model, build_iterator, load_checkpoint, accumulate_batch
from utils.model_utils import validate
from utils.loss_utils import LabelSmoothingLoss

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='device GPU/CPU')
parser.add_argument('--batch_size_trn', type=int, default=16, help='raw train batch size')
parser.add_argument('--batch_size_val', type=int, default=16, help='val/test batch size')
parser.add_argument('--batch_size_token', type=int, default=12000, help='train batch token number')
parser.add_argument('--gamma', type=float, default=2.0, help='')

parser.add_argument('--data_dir', type=str, default='data/uspto50k', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                    help='checkpoint directory')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint model file')

# graph
parser.add_argument('--pos_enc_dim', type=int, default=20, help='')
parser.add_argument('--pe_init', type=str, default='rand_walk', help='')
parser.add_argument('--g_L', type=int, default=4, help='')
parser.add_argument('--g_hidden_dim', type=int, default=128, help='')
parser.add_argument('--g_residual', type=bool, default=True, help='')
parser.add_argument('--g_edge_feat', type=bool, default=True, help='')
parser.add_argument('--g_readout', type=str, default='mean', help='')
parser.add_argument('--g_in_feat_dropout', type=float, default=0.0, help='')
parser.add_argument('--g_dropout', type=float, default=0.3, help='')
parser.add_argument('--g_batch_norm', type=bool, default=True, help='')
parser.add_argument('--g_use_lapeig_loss', type=bool, default=False, help='')
parser.add_argument('--g_alpha_loss', type=float, default=1e-4, help='')
parser.add_argument('--g_lambda_loss', type=float, default=1, help='')

####
parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--known_class', type=str, default='True', choices=['True', 'False'],
                    help='with reaction class known/unknown')
parser.add_argument('--shared_vocab', type=str, default='False', choices=['True', 'False'],
                    help='whether sharing vocab')
parser.add_argument('--shared_encoder', type=str, default='False', choices=['True', 'False'],
                    help='whether sharing encoder')
parser.add_argument('--max_epoch', type=int, default=5000, help='maximum epoch')
parser.add_argument('--max_step', type=int, default=5000000, help='maximum steps')
parser.add_argument('--lr_per_step', type=int, default=100000, help='validation steps frequency')
parser.add_argument('--report_per_step', type=int, default=100, help='train loss reporting steps frequency')
parser.add_argument('--save_per_step', type=int, default=10000, help='checkpoint saving steps frequency')
parser.add_argument('--val_per_step', type=int, default=10000, help='validation steps frequency')
parser.add_argument('--verbose', type=str, default=False, choices=['True', 'False'])

args = parser.parse_args()


def anneal_prob(step, k=2, total=150000):
    step = np.clip(step, 0, total)
    min_, max_ = 1, np.exp(k * 1)
    return (np.exp(k * step / total) - min_) / (max_ - min_)


def main(args):
    # log_path = os.path.join(args.proj_path, 'log')
    log_path = args.checkpoint_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_file_name = os.path.join(log_path, datetime.now().strftime("%D:%H:%M:%S").replace('/', ':') + '.txt')
    with open(log_file_name, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    train_iter, val_iter, vocab_itos_src, vocab_itos_tgt = \
        build_iterator(args, train=True, sample=False, augment=True)
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)
    global_step = 1
    if args.checkpoint:
        global_step, opt, model = load_checkpoint(args, model)
        global_step += 1
        lr = opt['param_groups'][0]['lr']
        eps = opt['param_groups'][0]['eps']
        optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
        print(f'the model learning rate is {lr}, eps is {eps}')
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    criterion_bond_rc = nn.BCELoss(reduction='sum')
    criterion_atom_rc = nn.BCELoss(reduction='sum')
    criterion_context_align = LabelSmoothingLoss(reduction='sum', smoothing=0.5)
    criterion_tokens = LabelSmoothingLoss(ignore_index=model.embedding_tgt.word_padding_idx,
                                          reduction='sum', apply_logsoftmax=False)

    loss_history_all, loss_history_token, loss_history_arc, loss_history_brc, loss_history_align, loss_history_center = [], [], [], [], [], []
    entry_count, src_max_length, tgt_max_length = 0, 0, 0
    true_batch = []

    if args.verbose == 'True':
        progress_bar = tqdm(train_iter)
    else:
        progress_bar = train_iter
    print('Begin:')
    for epoch in range(args.max_epoch):
        for batch in progress_bar:
            if global_step > args.max_step:
                print('Finish training.')
                break

            model.train()
            raw_src, raw_tgt = batch[0], batch[1]

            src_max_length = max(src_max_length, raw_src.shape[0])
            tgt_max_length = max(tgt_max_length, raw_tgt.shape[0])
            entry_count += raw_tgt.shape[1]

            if (src_max_length + tgt_max_length) * entry_count < args.batch_size_token:
                true_batch.append(batch)
            else:
                # Accumulate Batch
                src, tgt, gt_context_alignment, reacts_graph, product_graph, reacts_graph_num, product_graph_num, \
                    batch_rc_atoms, batch_rc_edges, pro_batch_adj, rea_batch_adj = accumulate_batch(true_batch)

                src, tgt, gt_context_alignment = src.to(args.device), tgt.to(args.device), gt_context_alignment.to(
                    args.device)
                del true_batch
                torch.cuda.empty_cache()

                generative_scores, atom_rc_scores, bond_rc_scores, context_scores, center_loss = \
                    model(src, tgt, product_graph, product_graph_num, pro_batch_adj, batch_rc_atoms)

                # Loss for language modeling:
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                gt_token_label = tgt[1:].view(-1)

                # Loss for atom-level reaction center identification:
                gt_atom_rc_label = np.array([])
                for j in range(len(product_graph_num[0])):
                    l = np.array([False] * product_graph_num[0][j])
                    l[batch_rc_atoms[j]] = True
                    gt_atom_rc_label = np.hstack((gt_atom_rc_label, l))
                gt_atom_rc_label = torch.Tensor(gt_atom_rc_label).to(args.device)
                pred_atom_rc_prob = atom_rc_scores.view(-1)

                # Loss for edge-level reaction center identification:
                gt_edge_rc_label = np.array([])
                for j in range(len(product_graph_num[1])):
                    l = np.array([False] * product_graph_num[1][j])
                    l[batch_rc_edges[j]] = True
                    gt_edge_rc_label = np.hstack((gt_edge_rc_label, l))
                gt_edge_rc_label = torch.Tensor(gt_edge_rc_label).to(args.device)
                bond_rc_scores = bond_rc_scores.view(-1)

                # Loss for context alignment:
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                # Compute all loss:
                loss_token = criterion_tokens(pred_token_logit, gt_token_label)
                loss_atom_rc = criterion_atom_rc(pred_atom_rc_prob, gt_atom_rc_label.float())
                loss_bond_rc = criterion_bond_rc(bond_rc_scores, gt_edge_rc_label.float())

                loss_context_align = 0
                # for context_score in context_scores:
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])

                loss_context_align += criterion_context_align(pred_context_align_logit, gt_context_align_label)

                loss = loss_token + loss_atom_rc + loss_bond_rc + loss_context_align + center_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history_all.append(loss.item())
                loss_history_token.append(loss_token.item())
                loss_history_arc.append(loss_atom_rc.item())
                loss_history_brc.append(loss_bond_rc.item())
                loss_history_align.append(loss_context_align.item())
                loss_history_center.append(center_loss.item())

                if global_step % args.lr_per_step == 0:
                    scheduler.step()

                if global_step % args.report_per_step == 0:
                    print_line = "[Time {} Epoch {} Iter {}] Loss {} NLL-Loss {} Rc-Loss {} {} Align-Loss {} Center-Loss {}".format(
                        datetime.now().strftime("%D:%H:%M:%S").replace('/', ':'), epoch, global_step,
                        round(np.mean(loss_history_all), 4), round(np.mean(loss_history_token), 4),
                        round(np.mean(loss_history_arc), 4), round(np.mean(loss_history_brc), 4),
                        round(np.mean(loss_history_align), 4), round(np.mean(loss_history_center), 4))
                    print(print_line)
                    with open(log_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')
                    loss_history_all, loss_history_token, loss_history_arc, loss_history_brc, loss_history_align, loss_history_center = [], [], [], [], [], []

                if global_step % args.save_per_step == 0:
                    checkpoint_path = args.checkpoint_dir + '/a_model_{}.pt'.format(global_step)
                    torch.save({'model': model.state_dict(), 'step': global_step, 'optim': optimizer.state_dict()},
                               checkpoint_path)
                    print('Checkpoint saved to {}'.format(checkpoint_path))

                if global_step % args.val_per_step == 0:
                    accuracy_arc, accuracy_brc, accuracy_token = \
                        validate(args, model, val_iter, model.embedding_tgt.word_padding_idx)
                    print_line = 'Validation accuracy: {} - {} - {}'.format(round(accuracy_arc, 4),
                                                                            round(accuracy_brc, 4),
                                                                            round(accuracy_token, 4))
                    print(print_line)
                    with open(log_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')

                # Restart Accumulation
                global_step += 1
                true_batch = [batch]
                entry_count, src_max_length, tgt_max_length = raw_src.shape[1], raw_src.shape[0], raw_tgt.shape[0]


if __name__ == '__main__':
    print(args)
    args.proj_path = path
    print(f"os.getcwd:{args.proj_path}")
    args.checkpoint_dir = os.path.join(args.proj_path, args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.intermediate_dir = os.path.join(args.proj_path, args.intermediate_dir)
    if not os.path.exists(args.intermediate_dir):
        os.makedirs(args.intermediate_dir, exist_ok=True)
    args.data_dir = os.path.join(args.proj_path, args.data_dir)
    main(args)
