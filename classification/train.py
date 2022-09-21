# train.py
import os
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from .architectures_class import ReceiverOnestep
from .dataset_class import Dataset
import numpy as np
import pickle
import random
import cv2
import wandb


USE_CUDA = torch.cuda.is_available()

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def save_image(receiver_input, gt, one_hot, log):
    for i in range(6):
        input = cv2.cvtColor((to_numpy(receiver_input[i, 0, ...].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
        target_idx = gt[i].cpu().numpy()
        choose_idx = one_hot[i].cpu().numpy()

        images = wandb.Image(input,
                             caption='target:' + str(int(target_idx)) + ' choice:' + str(int(choose_idx)))
        wandb.log({'classfication_img/' + str(i): images}, step=log)

max_precision = 0
def eval(opt, loader, receiver, steps, dreamer_step, save_img=True):
    global max_precision
    receiver.eval()

    total_cnt = 0.
    acc_cnt = 0.
    cnt = 0
    for data in loader:
        images, y = data

        images = images.cuda()
        y = y.cuda()

        with torch.no_grad():
            probs = receiver(images)

        _, amax = probs.max(dim=1)
        acc_cnt += (amax == y).sum()
        # print(list(zip(amax.cpu().numpy(), y.cpu().numpy())))
        total_cnt += y.shape[0]
        if cnt == 0:
            save_image(images, y, amax, dreamer_step)
        cnt += 1
    # print(acc_cnt/total_cnt)
    # exit(0)

    if acc_cnt/total_cnt > max_precision:
        model_save_name = os.path.join(opt.outf, f'classification_{dreamer_step}.pt')
        torch.save(receiver.state_dict(), model_save_name)
    max_precision = max(max_precision, acc_cnt/total_cnt)
    print('validate:{} acc:{} best_acc:{}'.format(steps, acc_cnt/total_cnt, max_precision))

    receiver.train()




def generate_data(opt, cate_list, dreamer_step):
    class2ID = {x: i for i, x in enumerate(cate_list)}
    cates = {}
    sketches = {}
    save_path = f'./classification_data_{dreamer_step}/' + opt.exp + '/'

    data_path = f'./to_cluster_{dreamer_step}_' + opt.exp
    with open('{}/img_name.p'.format(data_path), 'rb') as f:
        img_names = pickle.load(f)
    with open('{}/sketch.p'.format(data_path), 'rb') as f:
        imgs = pickle.load(f)

    old_name = None
    for iid, img_name in enumerate(img_names):
        if img_name != old_name:
            old_name = img_name
        else:
            continue
        cate = img_name.split('/')[0]
        if cate not in cates:
            cates[cate] = []
        if cate not in sketches:
            sketches[cate] = []

        cates[cate].append(img_name)
        sketches[cate].append(imgs[iid])

    train_img_names = []
    train_y = []
    test_img_names = []
    test_y = []
    train_imgs = []
    test_imgs = []

    for cate in cates.keys():
        train_img_names += cates[cate][:7]
        test_img_names += cates[cate][7:]
        train_y += [class2ID[cate]] * 7
        test_y += [class2ID[cate]] * len(cates[cate][7:])
        train_imgs += sketches[cate][:7]
        test_imgs += sketches[cate][7:]

    assert len(train_img_names) == len(train_y) == len(train_imgs)
    assert len(test_img_names) == len(test_y) == len(test_imgs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open('{}/train_img_names.p'.format(save_path), 'wb') as f:
        pickle.dump(train_img_names, f)
    with open('{}/test_img_names.p'.format(save_path), 'wb') as f:
        pickle.dump(test_img_names, f)
    with open('{}/train_y.p'.format(save_path), 'wb') as f:
        pickle.dump(train_y, f)
    with open('{}/test_y.p'.format(save_path), 'wb') as f:
        pickle.dump(test_y, f)
    with open('{}/train_imgs.p'.format(save_path), 'wb') as f:
        pickle.dump(train_imgs, f)
    with open('{}/test_imgs.p'.format(save_path), 'wb') as f:
        pickle.dump(test_imgs, f)

def train(opt, cate_list, dreamer_step):
    generate_data(opt, cate_list, dreamer_step)
    train_dataset = Dataset(opt.split_root, './classification_data/' + opt.exp + '/')

    loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=opt.batch_size, shuffle=True)
    test_dataset = Dataset(opt.split_root, './classification_data/' + opt.exp + '/', train=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=opt.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    receiver = ReceiverOnestep(device, opt.game_size, 128, opt, eps=opt.eps)

    receiver.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, receiver.parameters()),
        lr=opt.lr, betas=(opt.beta1, opt.beta2))
    loss = nn.CrossEntropyLoss()

    suffix = 'd_seed%d_clip%d_lr%d' \
    %(opt.manualSeed, opt.grad_clip,
        opt.lr)
    # added after
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    init_save_name = os.path.join(opt.outf,'players_init'+suffix)
    torch.save(receiver.state_dict(), init_save_name)
    # ENSURE THAT THEY HAVE THE SAME CURRICULUM AND Y
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    for i_games in range(701):
        total_loss = np.zeros(opt.batch_size)
        cnt = 0
        for data in loader:
            images, y = data
            images = images.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            probs = receiver(images)
            output = loss(probs, y)

            output.backward()
            optimizer.step()
            total_loss += output.detach().cpu().numpy()
            cnt += 1

        print('step:{} total_loss:{}'.format(i_games, (total_loss/cnt).mean()))
        # LR is decayed before the parameter update
        if i_games > opt.lr_decay_start and opt.lr_decay_start >= 0 and optimizer.param_groups[-1]['lr'] > 1e-4:
            frac = (i_games  - opt.lr_decay_start) / np.float32(opt.lr_decay_every)
            decay_factor =0.95**frac
            new_lr = opt.lr * decay_factor
            optimizer.param_groups[-1]['lr'] = new_lr

        if i_games % 10 == 0:
            eval(opt,
                 test_loader, receiver, i_games, dreamer_step, save_img=False)
    wandb.log({f'train_classification/acc': max_precision, }, step=dreamer_step)


def gradClamp(parameters, clip=1):
    for p in parameters:
        p.grad.data.clamp_(min=-clip,max=clip)



