import os
import sys

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path)
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
import copy
from utils.smiles_utils import *
from utils.translate_utils import translate_batch_original
from utils.build_utils import build_model, build_iterator, load_checkpoint

from tqdm import tqdm
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:3', help='device GPU/CPU')
parser.add_argument('--batch_size_val', type=int, default=16, help='batch size')
parser.add_argument('--beam_size', type=int, default=10, help='beam size')
parser.add_argument('--gamma', type=float, default=2.0, help='')

# graph
parser.add_argument('--pos_enc_dim', type=int, default=20, help='')
parser.add_argument('--pe_init', type=str, default='rand_walk', help='')
parser.add_argument('--g_L', type=int, default=4, help='')
parser.add_argument('--g_hidden_dim', type=int, default=128, help='')
parser.add_argument('--g_residual', type=bool, default=True, help='')
parser.add_argument('--g_edge_feat', type=bool, default=True, help='')
parser.add_argument('--g_readout', type=str, default='mean', help='')
parser.add_argument('--g_in_feat_dropout', type=float, default=0.0, help='')
parser.add_argument('--g_dropout', type=float, default=0.1, help='')
parser.add_argument('--g_batch_norm', type=bool, default=True, help='')
parser.add_argument('--g_use_lapeig_loss', type=bool, default=False, help='')
parser.add_argument('--g_alpha_loss', type=float, default=1e-4, help='')
parser.add_argument('--g_lambda_loss', type=float, default=1, help='')

parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--known_class', type=str, default='False', help='with reaction class known/unknown')
parser.add_argument('--shared_vocab', type=str, default='False', choices=['True', 'False'],
                    help='whether sharing vocab')
parser.add_argument('--shared_encoder', type=str, default='False', choices=['True', 'False'],
                    help='whether sharing encoder')

parser.add_argument('--data_dir', type=str, default='data/uspto50k', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                    help='checkpoint directory')
parser.add_argument('--checkpoint', type=str, default='unknown_model.pt', help='checkpoint model file')

args = parser.parse_args()

def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def translate(iterator, model, dataset):
    ground_truths = []
    generations = []
    invalid_token_indices = [dataset.tgt_stoi['<unk>']]
    # Translate:
    for index, batch in enumerate(tqdm(iterator, total=len(iterator))):
        src, tgt, _, _, _, _, _, _, _, _, _ = batch
        args.batch_index = index

        pred_tokens, pred_scores, atom_rc_scores, bond_rc_scores = translate_batch_original(args, model, batch,
                                                                                            beam_size=args.beam_size,
                                                                                            invalid_token_indices=invalid_token_indices)

        for idx in range(batch[0].shape[1]):
            gt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False))
            hypos = np.array([''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
            hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
            new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
            ordering = np.argsort(new_pred_score)[::-1]

            ground_truths.append(gt)
            generations.append(hypos[ordering])

    return ground_truths, generations


def main(args):
    # Build Data Iterator:
    iterator, dataset = build_iterator(args, train=False)

    # Load Checkpoint Model:
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    _, _, model = load_checkpoint(args, model)

    # Get Output Path:
    exp_version = 'typed' if args.known_class == 'True' else 'untyped'
    aug_version = '_augment' if 'augment' in args.checkpoint_dir else ''
    file_name = 'result/bs_top{}_generation_{}{}.pk'.format( args.beam_size, exp_version,
                                                                 aug_version)
    output_path = os.path.join(args.intermediate_dir, file_name)
    print('Output path: {}'.format(output_path))

    # Begin Translating:
    ground_truths, generations = translate(iterator, model, dataset)
    accuracy_matrix = np.zeros((len(ground_truths), args.beam_size))
    for i in range(len(ground_truths)):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1

    with open(output_path, 'wb') as f:
        pickle.dump((ground_truths, generations), f)

    for j in range(args.beam_size):
        print('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))


if __name__ == "__main__":
    print(args)
    set_random_seed(1)
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
