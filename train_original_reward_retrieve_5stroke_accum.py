# train.py
import os
import torch
print(torch.__version__)

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils_dreamer import parse_arguments
opt = parse_arguments()
if opt.setting == 'retrieve':
    from architectures_retrieve_5stroke import Sender, ReceiverOnestep, Players
else:
    from architectures_cum import Sender, ReceiverOnestep, Players
import pdb
from reinforce_one_stroke import *
from utils_dreamer import get_batch_random
from utils_dreamer import get_batch_random as get_batch_train_func
import numpy as np
import pickle
import random
import cv2
import classification.train as classification_train
from tqdm import tqdm
import wandb

wandb.login()
USE_CUDA = torch.cuda.is_available()

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def save_image_step(sender_input, receiver_input, gt, one_hot, res, step, log, save_path):
    for i in range(2):
        save_prefix = save_path + '/batch_' + str(i)
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix)
        for j in range(step[i]):
            canvas = cv2.cvtColor(res[i, j, ...].transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
            if j == step[i] - 1:
                input = cv2.cvtColor((to_numpy(sender_input[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                receiver_total = cv2.cvtColor(to_numpy(receiver_input[i, 0, ...].permute(1, 2, 0)), cv2.COLOR_BGR2RGB)
                for img_id in range(1, receiver_input.shape[1]):
                    receiver_total = np.concatenate(
                        [receiver_total, cv2.cvtColor(to_numpy(receiver_input[i, img_id, ...].permute(1, 2, 0)), cv2.COLOR_BGR2RGB)], axis=1)
                target_idx = gt[i].cpu().numpy()
                choose_idx = one_hot[i]
                receiver_total = cv2.cvtColor(receiver_total, cv2.COLOR_RGB2GRAY)
                images = wandb.Image(input.astype(np.uint8))
                wandb.log({str(i) + '/_sender_input.png': images}, step=log)
                images = wandb.Image(canvas)
                wandb.log({str(i) + '/_canvas.png': images}, step=log)
                images = wandb.Image(receiver_total.astype(np.uint8),
                                     caption='target:' + str(int(target_idx)) + ' choice:' + str(int(choose_idx)))
                wandb.log({str(i) + '/_receiver_input.png': images}, step=log)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

max_precision = 0
def eval(opt, players, steps, cate_list, save_img=True):
    global max_precision
    players.sender.eval()
    players.receiver.eval()

    total_loss = torch.zeros(opt.batch_size)
    total_acc = torch.zeros(opt.batch_size)
    to_save_list = ['giraffe', 'deer', 'rabbit', 'horse', 'pig', 'snail', 'camel', 'sheep', 'cow', 'elephant']
    for i_game in range(opt.validate_episode):
        y, images_vectors_sender, images_vectors_receiver, batch_cate_list = get_batch_train_func(opt, cate_list)

        if opt.cuda:
            y = Variable(y.cuda())
        else:
            y = Variable(y)

        with torch.no_grad():
            loss, res, acc, step, choice, _ = players(images_vectors_sender,
                images_vectors_receiver, opt, y, train=False)

        save_image_step_evolve((res * 255).astype(np.uint8), step, steps, to_save_list, batch_cate_list)
        # print acc
        total_loss += loss.cpu()
        total_acc += acc

    total_loss /= opt.validate_episode
    total_acc /= opt.validate_episode
    wandb.log({'val/loss': total_loss.numpy().mean(),
               'val/precision': total_acc.numpy().mean(),
               }, step=steps)
    max_precision = max(max_precision, total_acc.numpy().mean())
    print('validate:{} total_loss:{} acc:{} best_acc:{}'.format(steps, total_loss.numpy().mean(), total_acc.numpy().mean(), max_precision))
    # exit(0)
    if not opt.sender_fixed:
        players.sender.train()
    players.receiver.train()
    players.sender.apply(set_bn_eval)

def eval_by_step(opt, players, steps, cate_list, max_step):
    players.sender.eval()
    players.receiver.eval()
    width = 128

    total_acc = torch.zeros(opt.batch_size)
    for i_game in range(opt.validate_episode):
        y, images_vectors_sender, images_vectors_receiver, _ = get_batch_train_func(opt, cate_list)

        if opt.cuda:
            y = Variable(y.cuda())
        else:
            y = Variable(y)

        ## TODO
        canvas = torch.zeros([opt.batch_size, 3, width, width], requires_grad=False).cuda()
        canvas0 = torch.zeros([opt.batch_size, 3, width, width], requires_grad=False).cuda()
        mask = torch.ones(opt.batch_size).long().cuda()
        precision = np.zeros(opt.batch_size)
        step_num = -1
        for i in range(max_step):
            canvas0 = canvas.detach()
            canvas = sender_action(players.sender,
                                   images_vectors_sender, i, canvas0, num_stroke=opt.num_stroke)
            step_num += 1

        one_hot_output, receiver_probs = receiver_action_retrieve(players.receiver,
                                                                  images_vectors_receiver, canvas0, canvas, opt,
                                                                  is_train=False)
        _, amax = one_hot_output.max(dim=1)
        next_mask = mask * (amax == opt.game_size - 1)
        mask_ = mask.cpu().numpy().astype(np.bool)
        next_mask_ = next_mask.cpu().numpy().astype(np.bool)
        precision[mask_ * ~next_mask_] += (y == amax).float().cpu().numpy()[mask_ * ~next_mask_]

        # print acc
        total_acc += precision

    total_acc /= opt.validate_episode

    wandb.log({
               f'val/precision_{max_step}': total_acc.numpy().mean(),
               }, step=steps)

    # exit(0)
    if not opt.sender_fixed:
        players.sender.train()
    players.receiver.train()
    players.sender.apply(set_bn_eval)

def save_step(opt, players, steps, cate_list):
    players.sender.eval()
    players.receiver.eval()

    save_path = opt.outf + 'imgs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    y, images_vectors_sender, images_vectors_receiver, _ = get_batch_train_func(opt, cate_list)

    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    with torch.no_grad():
        loss, res, acc, step, choice, _ = players(images_vectors_sender,
            images_vectors_receiver, opt, y, train=False)

    save_image_step(images_vectors_sender, images_vectors_receiver, y, choice, (res*255).astype(np.uint8), step, steps, save_path)

    if not opt.sender_fixed:
        players.sender.train()
    players.receiver.train()
    players.sender.apply(set_bn_eval)

def get_batch_random_evolve(opt, pair_list):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, opt.game_size-1, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    width = 128
    img_names = []
    for i in range(len(pair_list)):
        pair_names = pair_list[i].split()
        distractors = pair_names[:opt.game_size - 1]
        target = int(pair_names[opt.game_size])
        imgs = torch.zeros(opt.game_size - 1, 3, 128, 128)
        for j in range(opt.game_size - 1):
            img_name = distractors[j]
            img_name = img_name.split('_ske')[0] + '_img.jpg'
            img = cv2.imread(opt.data_root + img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (width, width))
            img = img.reshape(1, width, width, 3)
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.tensor(img).float()
            imgs[j, ...] = img

        img_name = pair_names[opt.game_size - 1]
        img_names.append(img_name)
        img = cv2.imread(opt.data_root + img_name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (width, width))
        img = img.reshape(1, width, width, 3)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.tensor(img).float()
        img_s = img

        if opt.cuda:
            img_s = Variable(img_s.cuda())
            imgs = Variable(imgs.cuda())
        else:
            img_s = Variable(img_s)
            imgs = Variable(imgs)
        images_vectors_sender[i, ...] = img_s
        images_vectors_receiver[i, ...] = imgs
        y[i] = target

    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    return y, images_vectors_sender, images_vectors_receiver, img_names

def save_image_step_evolve(res, step, log, to_save_list, batch_cate_list):
    for i in range(len(res)):
        img_name = batch_cate_list[i]
        cate = img_name.split('/')[-2]
        if cate in to_save_list:
            canvas = cv2.cvtColor(res[i, step[i]-1, ...].transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
            images = wandb.Image(canvas)
            wandb.log({cate + '/' + img_name.split('/')[-1]: images}, step=log)


def save_step_evolve(opt, players, steps, pair_list):
    players.sender.eval()
    players.receiver.eval()

    cate_names = []
    sketches = []

    for i in range(0, len(pair_list), opt.batch_size):
        y, images_vectors_sender, images_vectors_receiver, img_names = get_batch_random_evolve(opt, pair_list[i:i+opt.batch_size])

        if opt.cuda:
            y = Variable(y.cuda())
        else:
            y = Variable(y)

        with torch.no_grad():
            loss, res, acc, step, choice, _ = players(images_vectors_sender,
                images_vectors_receiver, opt, y, train=False)

        for j in range(len(img_names)):
            img_name = img_names[j]
            canvas = res[j, step[j] - 1, ...]
            sketches.append(canvas)
            cate_names.append(img_name)

    save_path = './to_cluster_' + opt.exp + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + 'sketch.p', 'wb') as f:
        pickle.dump(sketches, f)
    with open(save_path + 'img_name.p', 'wb') as f:
        pickle.dump(cate_names, f)

    if not opt.sender_fixed:
        players.sender.train()
    players.receiver.train()
    players.sender.apply(set_bn_eval)

def test_generalization(opt, players, steps, pair_list, set_name):
    players.sender.eval()
    players.receiver.eval()

    gt_test = []
    game_steps = []

    for i in tqdm(range(0, len(pair_list), opt.batch_size)):
        y, images_vectors_sender, images_vectors_receiver, img_names = get_batch_random_evolve(opt, pair_list[i:i+opt.batch_size])
        t = len(img_names)
        if opt.cuda:
            y = Variable(y.cuda())
        else:
            y = Variable(y)

        with torch.no_grad():
            loss, res, acc, step, choice, _ = players(images_vectors_sender,
                images_vectors_receiver, opt, y, train=False)
        gt_test += list(acc[:t])
        game_steps += list(step[:t])

    assert len(game_steps) == len(pair_list)
    assert len(gt_test) == len(pair_list)
    print('avg_step', sum(game_steps) / float(len(pair_list)))
    print('test_acc', sum(gt_test) / float(len(pair_list)))
    wandb.log({f'generalization/test_{set_name}': sum(gt_test) / float(len(pair_list)),
               f'generalization/avg_step_{set_name}': sum(game_steps) / float(len(pair_list)),
               }, step=steps)

    if not opt.sender_fixed:
        players.sender.train()
    players.receiver.train()
    players.sender.apply(set_bn_eval)

def train(opt):

    with open(opt.category_list, 'r') as f:
        cate_list = f.readlines()[:-1]

    cate_list = [x[:-1] for x in cate_list]

    with open(opt.split_root + '_train.txt', 'r') as f:
        pair_list = f.readlines()

    with open(opt.split_root + '_unseen.txt', 'r') as f:
        pair_list_test = f.readlines()
    with open(opt.split_root + '_unseen_cate.txt', 'r') as f:
        pair_list_unseen_cate = f.readlines()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sender = Sender(device, opt.max_step, opt, width = 128)
    receiver = ReceiverOnestep(device, opt.game_size, 128, opt)

    sender.to(device)
    receiver.to(device)
    players = Players(sender, receiver)
    players.sender.apply(set_bn_eval)
    if opt.sender_fixed:
        players.sender.eval()

    if opt.cuda:
        players.cuda()

    if opt.opti == 'adam':
        optimizer_r = optim.Adam(filter(lambda p: p.requires_grad, players.receiver.parameters()),
            lr=opt.receiver_lr, betas=(opt.beta1, opt.beta2))
        optimizer_s = optim.Adam(filter(lambda p: p.requires_grad, players.sender.parameters()),
                               lr=0., betas=(opt.beta1, opt.beta2))
    elif opt.opti == 'sgd':
        optimizer = optim.SGD(players.parameters(),
            lr=opt.lr, momentum=0.0, dampening=0, weight_decay=0,
            nesterov=False)

    suffix = 'd_seed%d_clip%d_lr%.4f' \
    %(opt.manualSeed, opt.grad_clip,
        opt.lr)
    # added after
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    init_save_name = os.path.join(opt.outf,'players_init'+suffix)
    torch.save(players.state_dict(), init_save_name)
    # ENSURE THAT THEY HAVE THE SAME CURRICULUM AND Y
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    for i_games in range(opt.start_step, opt.start_step + opt.n_games+1):
        y, images_vectors_sender, images_vectors_receiver, _ = get_batch_random(opt, cate_list)

        optimizer_s.zero_grad()
        optimizer_r.zero_grad()
        total_loss, res, _, cnt, _, actual_reward =\
            players(images_vectors_sender, images_vectors_receiver, opt, y)

        if np.any(np.isnan(total_loss.data.clone().cpu().numpy())):
            pdb.set_trace()


        total_loss.mean().backward()

        if opt.grad_clip:
            gradClamp(players.receiver.parameters())

        wandb.log({'train/loss': total_loss.data.clone().cpu().numpy().mean(),
                   'train/loss_std': total_loss.data.clone().cpu().numpy().std(),
                   'train/actual_reward': actual_reward.data.clone().cpu().numpy().mean(),
                   'train/avg_step': cnt.mean(),
                   'train/sender_lr': optimizer_s.param_groups[-1]['lr'],
                   'train/receiver_lr': optimizer_r.param_groups[-1]['lr'],
                   }, step=i_games)
        print('step:{} total_loss:{} avg_step:{}'.format(i_games, total_loss.data.clone().cpu().numpy().mean(), cnt.mean()))
        # LR is decayed before the parameter update
        if i_games <= opt.lr_decay_start:
            new_lr = opt.sender_lr / opt.lr_decay_start * i_games
            optimizer_s.param_groups[-1]['lr'] = new_lr

        if i_games > opt.lr_decay_start and opt.lr_decay_start >= 0 and optimizer_r.param_groups[-1]['lr'] > 5e-6:
            frac = (i_games - opt.lr_decay_start) / np.float32(opt.lr_decay_every)
            decay_factor = opt.receiver_decay ** frac
            new_lr = opt.receiver_lr * decay_factor
            optimizer_r.param_groups[-1]['lr'] = new_lr
            decay_factor = opt.sender_decay ** frac
            new_lr = opt.sender_lr * decay_factor
            optimizer_s.param_groups[-1]['lr'] = new_lr

        optimizer_r.step()
        optimizer_s.step()

        if i_games % 100 == 0:
            save_step(opt, players, i_games, cate_list)

        if i_games % 100 == 0:
             eval(opt,
                     players, i_games, cate_list)
             for j in range(1, 8, 2):
                 eval_by_step(opt, players, i_games, cate_list, j)

        if i_games % 5000 == 0:
            save_step_evolve(opt, players, i_games, pair_list)
            test_generalization(opt, players, i_games, pair_list, 'train')
            test_generalization(opt, players, i_games, pair_list_test, 'test')
            test_generalization(opt, players, i_games, pair_list_unseen_cate, 'unseen_cate')
            classification_train.train(opt, cate_list, i_games)

        if i_games % 1000 == 0:
            # save current model
            model_save_name = os.path.join(opt.outf,'players' +
                                    suffix + '_i%d.pt'%i_games)
            torch.save(players.state_dict(), model_save_name)

    model_save_name = os.path.join(opt.outf,'players'+suffix)
    torch.save(players.state_dict(), model_save_name)


def gradClamp(parameters, clip=1):
    for p in parameters:
        p.grad.data.clamp_(min=-clip,max=clip)

if __name__ == "__main__":

    with wandb.init(project=opt.log_outf, name=opt.exp, entity='pictionary', config=opt):
        train(wandb.config)
