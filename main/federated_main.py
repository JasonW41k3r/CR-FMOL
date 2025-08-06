#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os, copy, time, pickle, numpy as np, torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNCifar, Adult
from utils  import (get_dataset, exp_details,
                    solve_centered_w, solve_capped_w, solve_cone_w)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
NUM_CLIENTS_EXPECTED = 100          # plot_fig3a 最右维

def flat(sd):
    """flatten state-dict to 1-D numpy"""
    return torch.cat([v.view(-1) for v in sd.values()]).cpu().double().numpy()

def pad_or_cut(arr, L=NUM_CLIENTS_EXPECTED):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < L:
        arr = np.concatenate([arr, np.full(L-len(arr), np.nan, np.float32)])
    elif len(arr) > L:
        arr = arr[:L]
    return arr

def main():
    start = time.time()
    args = args_parser()
    logger = SummaryWriter('../logs')
    exp_details(args)

    # reproducibility & device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # ---------- data ----------
    train_ds, test_ds, user_groups = get_dataset(args)

    # ---------- model ----------
    if args.model == 'cnn':
        global_model = CNNMnist(args) if args.dataset == 'mnist' else CNNCifar(args)
    elif args.model == 'mlp':
        shp = train_ds[0][0].shape
        global_model = MLP(int(np.prod(shp)), 64, args.num_classes)
    elif args.model == 'lr' and args.dataset == 'adult':
        global_model = Adult()
    else:
        raise ValueError('Unrecognized model / dataset')
    global_model.to(device).train()

    # ---------- containers ----------
    loss_mean, loss_min, loss_max, acc_hist = [], [], [], []
    user_test_acc, user_test_loss, total_accuracy = [], [], []
    server_lr_curve = []

    old_w = global_model.state_dict()
    eta, decay, print_every = args.global_lr, args.global_lr_decay, 10

    # ---------- training ----------
    for epoch in tqdm(range(args.epochs), desc='Global'):
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs = np.random.choice(range(args.num_users), m, replace=False)

        local_ws, local_ls, local_ns = [], [], []
        for uid in idxs:
            loc = LocalUpdate(args, train_ds, user_groups[uid], logger)
            dw, l, *_ , nrm = loc.update_weights(copy.deepcopy(global_model), epoch)
            local_ws.append(dw); local_ls.append(l); local_ns.append(nrm)

        # coeffs
        n = len(local_ws)
        if args.method == 'cpf':
            coeffs = solve_cone_w([flat(dw) for dw in local_ws], lam_deg=args.cone_angle)
        elif args.cap < 1:
            coeffs = solve_capped_w(local_ws, C=args.cap)
        elif args.epsilon != 0:
            coeffs = solve_centered_w(local_ws, epsilon=args.epsilon)
        else:
            coeffs = np.ones(n) / n

        # server lr
        lr_s = eta*(decay**(epoch//100)) if decay <= 1 else (np.median(local_ns) if decay == 2 else eta)
        server_lr_curve.append(lr_s)

        # aggregate
        new_w = copy.deepcopy(old_w)
        for k in new_w:
            for i in range(n):
                new_w[k] += lr_s * coeffs[i] * local_ws[i][k]
        global_model.load_state_dict(new_w)

        # per-round metrics
        mean_l, min_l, max_l = map(float, (np.mean(local_ls), np.min(local_ls), np.max(local_ls)))
        loss_mean.append(mean_l); loss_min.append(min_l); loss_max.append(max_l)

        accs = []
        global_model.eval()
        for uid in range(args.num_users):
            loc = LocalUpdate(args, train_ds, user_groups[uid], logger)
            a, _ = loc.inference(global_model); accs.append(a)
        acc_hist.append(sum(accs)/len(accs))

        # tensorboard
        logger.add_scalar('ClientLoss/Mean', mean_l, epoch)
        logger.add_scalar('ClientLoss/Min',  min_l,  epoch)
        logger.add_scalar('ClientLoss/Max',  max_l,  epoch)
        logger.add_scalar('Train/Acc',       acc_hist[-1], epoch)
        logger.add_scalar('Server/LR',       lr_s,         epoch)

        # every 10 rounds: collect user test curves
        if (epoch + 1) % print_every == 0:
            te_acc, _ = test_inference(args, global_model, test_ds)
            total_accuracy.append(te_acc)

            accs_pad  = pad_or_cut(accs)
            loss_pad  = pad_or_cut(local_ls)
            user_test_acc.append(accs_pad)
            user_test_loss.append(loss_pad)

    logger.close()

    # ---------- save ----------
    tmpl = ("Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]"
            "_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}]")
    base = tmpl.format(args.epochs, args.frac, args.local_ep, args.local_bs,
                       args.lr, args.global_lr, args.momentum, args.model, args.normalize,
                       args.epsilon, args.dataset, int(args.iid), args.seed,
                       args.global_lr_decay, args.prox_weight, args.qffl,
                       args.Lipschitz_constant)
    os.makedirs('save/objects', exist_ok=True)

    # pickle
    with open(os.path.join('save/objects', base + '.pickle'), 'wb') as f:
        pickle.dump((loss_mean, acc_hist), f)

    # npz
    user_acc_arr  = np.asarray(user_test_acc,  dtype=np.float32)   # (T/10,100)
    user_loss_arr = np.asarray(user_test_loss, dtype=np.float32)   # (T/10,100)
    acc_arr       = np.asarray(total_accuracy, dtype=np.float32)   # (T/10,)
    np.savez(os.path.join('save/objects', base + '.npz'),
             acc=acc_arr,
             user=(user_acc_arr, user_loss_arr))

    print('Saved legacy pickle & npz for plotting.')
    print('Runtime: %.1f min' % ((time.time()-start)/60))

if __name__ == '__main__':
    main()
