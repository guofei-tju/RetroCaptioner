from rdkit import RDLogger, Chem

RDLogger.DisableLog('rdApp.*')

import argparse
import torch
print(torch.cuda.is_available())
import logging
import time
import pickle
from retro_star.common import prepare_starting_molecules, \
    prepare_molstar_planner, smiles_to_fp, args
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from inference import OneStepRetroModel
import os

def prepare_single_step_model(model_path):
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

class RSPlanner:
    def __init__(self,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=True,
                 starting_molecules="./retro_star/dataset/origin_dict.csv",
                 model_dump="./used3d_no_contras_checkpoint_typed/a_model_510000.pt",
                 save_folder="data/multi-step/retro_data/saved_models",
                 value_model="./retro_star/saved_models/best_epoch_final_4.pt",
                 fp_dim=2048,
                 viz=False,
                 viz_dir='viz'):

        setup_logger()
        device = torch.device('cuda:%d' % gpu if gpu >= 0 else 'cpu')
        args.device = device
        starting_mols = prepare_starting_molecules(starting_molecules)

        one_step = prepare_single_step_model(model_dump)

        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model_f = os.path.join(args.proj_path, value_model)
            logging.info('Loading value nn from %s' % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x: 0.

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )
        self.starting_mols = starting_mols

    def plan(self, target_mol, need_action=False):
        t0 = time.time()
        flag_removed = False
        if target_mol in self.starting_mols:
            flag_removed = True
            self.starting_mols.remove(target_mol)
        succ, msg = self.plan_handle(target_mol, self.starting_mols)

        if flag_removed:
            self.starting_mols.add(target_mol)
        if succ:
            result = {
                'succ': succ,
                'time': time.time() - t0,
                'iter': msg[1],
                'routes': msg[0].serialize(need_action=need_action),
                'route_cost': msg[0].total_cost,
                'route_len': msg[0].length
            }
            return result

        else:
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None



if __name__ == '__main__':
    print(args)
    args.proj_path = os.path.dirname(os.path.realpath(__file__))
    args.checkpoint_dir = os.path.join(args.proj_path, args.checkpoint_dir)
    args.intermediate_dir = os.path.join(args.proj_path, args.intermediate_dir)
    model_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    planner = RSPlanner(
        gpu=args.gpu,
        use_value_fn=args.use_value_fn,
        model_dump = model_path,
        iterations=args.iterations,
        expansion_topk=args.expansion_topk,
        viz=args.viz,
    )
    result = planner.plan(args.smiles)
    print(result)
