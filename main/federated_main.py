#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os, copy, time, pickle, numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNCifar, Adult
from utils import (get_dataset, exp_details,
                   solve_centered_w, solve_capped_w, solve_cone_w)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def flatten_state_dict(sd):
    return torch.cat([v.view(-1) for v in sd.values()]).cpu().double().numpy()


def main():
    start = time.time()
    args = args_parser()
    log_dir = '../logs'
    logger = SummaryWriter(log_dir)
    exp_details(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # -------- data & model --------
    train_ds, test_ds, user_groups = get_dataset(args)

    if args.model == 'cnn':
        global_model = CNNMnist(args) if args.dataset == 'mnist' else CNNCifar(args)
    elif args.model == 'mlp':
        img_shape = train_ds[0][0].shape
        global_model = MLP(int(np.prod(img_shape)), 64, args.num_classes)
    elif args.model == 'lr' and args.dataset == 'adult':
        global_model = Adult()
    else:
        raise ValueError('bad model/dataset')

    global_model.to(device).train()
    print('trainable params:', sum(p.numel() for p in global_model.parameters() if p.requires_grad))

    # -------- logs --------
    loss_mean_hist, loss_min_hist, loss_max_hist, acc_hist = [], [], [], []
    test_loss_curve, test_acc_curve, server_lr_curve = [], [], []

    old_w = global_model.state_dict()
    eta, decay, print_every = args.global_lr, args.global_lr_decay, 10

    for epoch in tqdm(range(args.epochs), desc='Global'):
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs = np.random.choice(range(args.num_users), m, replace=False)

        local_ws, local_ls, local_ns = [], [], []
        for uid in idxs:
            loc = LocalUpdate(args, train_ds, user_groups[uid], logger)
            dw, l, _, _, nrm = loc.update_weights(copy.deepcopy(global_model), epoch)
            local_ws.append(dw); local_ls.append(l); local_ns.append(nrm)

        # coeffs
        n = len(local_ws)
        if args.method == 'cpf':
            grads = [flatten_state_dict(dw) for dw in local_ws]
            coeffs = solve_cone_w(grads, lam_deg=args.cone_angle)
        elif args.cap < 1:
            coeffs = solve_capped_w(local_ws, C=args.cap)
        elif args.epsilon != 0:
            coeffs = solve_centered_w(local_ws, epsilon=args.epsilon)
        else:
            coeffs = np.ones(n)/n

        # lr_server
        lr_s = eta*(decay**(epoch//100)) if decay<=1 else (np.median(local_ns) if decay==2 else eta)
        server_lr_curve.append(lr_s)

        # aggregate
        new_w = copy.deepcopy(old_w)
        for k in new_w:
            for i in range(n):
                new_w[k] += lr_s*coeffs[i]*local_ws[i][k]
        global_model.load_state_dict(new_w)

        # metrics
        mean_l, min_l, max_l = float(np.mean(local_ls)), float(np.min(local_ls)), float(np.max(local_ls))
        loss_mean_hist.append(mean_l); loss_min_hist.append(min_l); loss_max_hist.append(max_l)

        global_model.eval(); accs=[]
        for uid in range(args.num_users):
            loc=LocalUpdate(args, train_ds, user_groups[uid], logger)
            acc,_=loc.inference(global_model); accs.append(acc)
        avg_acc = sum(accs)/len(accs); acc_hist.append(avg_acc)

        # tb
        logger.add_scalar('ClientLoss/Mean', mean_l, epoch)
        logger.add_scalar('ClientLoss/Min',  min_l,  epoch)
        logger.add_scalar('ClientLoss/Max',  max_l,  epoch)
        logger.add_scalar('Train/Acc', avg_acc, epoch)
        logger.add_scalar('Server/LR', lr_s, epoch)

        if (epoch+1)%print_every==0:
            te_acc, te_loss = test_inference(args, global_model, test_ds)
            test_acc_curve.append(te_acc); test_loss_curve.append(te_loss)
            print(f'[Eval] ep{epoch+1} TestAcc {100*te_acc:.2f}')

    logger.close()

    final_acc, final_loss = test_inference(args, global_model, test_ds)

    # --- save new-format metrics ---
    os.makedirs('save/metrics', exist_ok=True)
    with open(f"save/metrics/{args.method}_{args.dataset}_{args.epochs}r.pkl","wb") as f:
        pickle.dump({
            'loss_mean':loss_mean_hist,'loss_min':loss_min_hist,'loss_max':loss_max_hist,
            'train_acc':acc_hist,'test_loss_curve':test_loss_curve,'test_acc_curve':test_acc_curve,
            'server_lr_curve':server_lr_curve,'final_test_acc':float(final_acc),
            'final_test_loss':float(final_loss),'args':vars(args)},f)

    # --- Compat pickle for legacy plotting ---
    compat_name = "Ep[{}]_frac[{}]_{}_{}_{}_{}_Mom[{}]_{}_N[{}]_EPS[{}]_{}_iid[{}]_seed[{}]_decay[{}]_prox[{}]_qffl[{}]_Lipsch[{}].pickle".format(
        args.epochs,  # 2000
        args.frac,  # 0.1
        args.local_ep,  # 1
        args.local_bs,  # 10
        args.lr,  # 0.01
        args.global_lr,  # 1.0
        args.momentum,  # 0.0
        args.model,  # 'cnn'
        args.normalize,  # 0
        args.epsilon,  # *关键*：0/0.05/0.1/0.5/1.0
        args.dataset,  # 'cifar'
        int(args.iid),  # 1
        args.seed,  # 1–5
        args.global_lr_decay,  # 1.0
        args.prox_weight,  # 0.0
        args.qffl,  # 0.0
        args.Lipschitz_constant  # 1.0
    )
    os.makedirs('save/objects', exist_ok=True)
    with open(os.path.join('save/objects',compat_name),'wb') as f2:
        # legacy code expects (loss_series, acc_series) with acc in 0-1
        pickle.dump((loss_mean_hist, acc_hist), f2)

    print('Compat pickle saved →', compat_name)
    print('Total runtime %.1f min' % ((time.time()-start)/60))

if __name__ == '__main__':
    main()