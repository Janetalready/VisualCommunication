import os
import random
import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import pdb
import glob
import cv2
from typing import Iterable
from torch.nn import Module

def compute_similarity_images(space):
    normalized=space/(torch.norm(space,p=2,dim=1,keepdim=True))
    pairwise_cosines_matrix=torch.matmul(normalized,normalized.t())
    return pairwise_cosines_matrix[0,1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split_root', default='./data/same_cate_mul_image300', help='split root folder')
    parser.add_argument(
        '--category_list', default='./data/category.txt', help='split root folder')
    parser.add_argument(
        '--data_root', default='./data/output/', help='data root folder')
    parser.add_argument(
        '--sender_path', default='../ddpg_comm/pretrained/actor_one_stroke.pkl', help='pretrained folder')
    parser.add_argument(
        '--resume_path', default=None, help='pretrained folder')
    parser.add_argument('--max_step', type=int,
                        help='number of drawing steps', default=10)
    parser.add_argument('--num_stroke', type=int,
                        help='number of strokes', default=3)
    parser.add_argument('--sender_decay', type=float,
                        help='sender_decay_rate', default=0.95)
    parser.add_argument('--receiver_decay', type=float,
                        help='receiver_decay_rate', default=0.95)
    parser.add_argument('--sender_fixed', type=int,
                                    help='fix sender', default=0)
    parser.add_argument('--sender_add_noise', type=int,
                        help='add noise to sender output', default=0)
    parser.add_argument('--sender_fix_resnet', type=int,
                        help='add noise to sender output', default=0)
    parser.add_argument('--sender_fix_norm', type=int,
                        help='add noise to sender output', default=1)
    parser.add_argument('--start_step', type=int,
                                    help='start steps', default=0)
    parser.add_argument('--validate_episode', type=int,
                        help='number of validation games', default=32)
    parser.add_argument('--step_cost', type=int,
                        help='step cost', default=0)
    parser.add_argument('--discount', type=float,
                        help='temporal decay', default=0.9)
    parser.add_argument('--lambda_', type=float,
                        help='horizon weight', default=0.95)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate, default=0.01')
    parser.add_argument('--value_lr', type=float, default=0.0001,
                        help='learning rate, default=0.01')
    parser.add_argument('--receiver_lr', type=float, default=0.0001,
                        help='learning rate, default=0.01')
    parser.add_argument('--sender_lr', type=float, default=0.0001,
                        help='learning rate, default=0.01')
    parser.add_argument('--lr_decay_start', type=int, default=2000,
                        help='learning rate decay iter start, default=10000')
    parser.add_argument('--lr_decay_every', type=int, default=1000,
                        help='every how many iter thereafter to div LR by 2, default=5000')
    parser.add_argument('--opti', type=str, default='adam',
                        help='optimizer, default=adam')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for adam. default=0.8')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for adam. default=0.999')
    parser.add_argument('--outf', default='./output_retrieve_sketch1/',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--exp', default='change_game_size_4_test',
                        help='folder to experiment')
    parser.add_argument('--log_outf', default='train_log_dreamer',
                                    help='folder to training log')
    parser.add_argument('--manualSeed', type=int,default=0,
                        help='manual seed')
    parser.add_argument('--game_size', type=int, default=4,
                        help='game size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--n_games', type=int, default=50000,
                        help='number of games')
    parser.add_argument('--grad_clip', type=int, default=0,
                        help='gradient clipping')
    opt = parser.parse_args()

    if opt.outf == '.':
        if os.environ.get('SLURM_JOB_DIR') is not None:
            opt.outf = os.environ.get('SLURM_JOB_DIR')

    if os.environ.get('SLURM_JOB_ID') is not None:
        opt.job_id = os.environ.get('SLURM_JOB_ID')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    return opt

def get_batch(opt, loader, cnt, test=False):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, 4, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    if test:
        batch_inds = np.random.choice(len(loader.dataset), opt.batch_size, replace=False)
    else:
        batch_inds = range(cnt*opt.batch_size, min((cnt+1)*opt.batch_size, 1000))
    # print(batch_inds)
    for iid, ind in enumerate(batch_inds):
        img_s, target, imgs = loader.dataset[ind]
        if opt.cuda:
            img_s = Variable(img_s.cuda())
            imgs = Variable(imgs.cuda())
        else:
            img_s = Variable(img_s)
            imgs = Variable(imgs)
        images_vectors_sender[iid, ...] = img_s
        images_vectors_receiver[iid, ...] = imgs
        y[iid] = target

    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    return y, images_vectors_sender, images_vectors_receiver

def get_batch_random(opt, category_list):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, opt.game_size-1, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    width = 128
    cate_list = []
    # np.random.seed(opt.manualSeed)
    for i in range(opt.batch_size):
        cates = np.random.choice(category_list, opt.game_size-1, replace=False)
        target = np.random.randint(opt.game_size-1)
        imgs = torch.zeros(opt.game_size-1, 3, 128, 128)
        for j in range(opt.game_size-1):
            img_names = sorted(glob.glob(os.path.join(opt.data_root, cates[j], '*_img.jpg')))[:10]
            if j == target:
                img_name = np.random.choice(img_names, 1)[0]
                img_name = img_name.split('_img')[0] + '_ske.jpg'
                img = cv2.imread(img_name, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (width, width))
                img = img.reshape(1, width, width, 3)
                img = np.transpose(img, (0, 3, 1, 2))
                img = torch.tensor(img).float()
                img_s = img
                cate_list.append(img_name)

            img_name = np.random.choice(img_names, 1)[0]
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (width, width))
            img = img.reshape(1, width, width, 3)
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.tensor(img).float()
            imgs[j, ...] = img

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

    return y, images_vectors_sender, images_vectors_receiver, cate_list

def get_batch_random_eval(opt, category_list):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, opt.game_size-1, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    width = 128
    np.random.seed(opt.manualSeed)
    for i in range(opt.batch_size):
        cates = np.random.choice(category_list, opt.game_size-1, replace=False)
        target = np.random.randint(opt.game_size-1)
        imgs = torch.zeros(opt.game_size-1, 3, 128, 128)
        for j in range(opt.game_size-1):
            img_names = sorted(glob.glob(os.path.join(opt.data_root, cates[j], '*_img.jpg')))[:10]
            if j == target:
                img_name = np.random.choice(img_names, 1)[0]
                img_name = img_name.split('_img')[0] + '_ske.jpg'
                img = cv2.imread(img_name, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (width, width))
                img = img.reshape(1, width, width, 3)
                img = np.transpose(img, (0, 3, 1, 2))
                img = torch.tensor(img).float()
                img_s = img

            img_name = np.random.choice(img_names, 1)[0]
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (width, width))
            img = img.reshape(1, width, width, 3)
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.tensor(img).float()
            imgs[j, ...] = img

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

    return y, images_vectors_sender, images_vectors_receiver

def get_batch_random_evolve(opt, pair_list):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, opt.game_size-1, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    width = 128
    np.random.seed(opt.manualSeed)
    for i in range(opt.batch_size):
        pair_names = pair_list[i].split()
        distractors = pair_names[:3]
        target = int(pair_names[4])
        imgs = torch.zeros(opt.game_size-1, 3, 128, 128)
        for j in range(opt.game_size-1):
            img_name = distractors[j]
            img_name = img_name.split('_ske')[0] + '_img.jpg'
            img = cv2.imread(opt.data_root + img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (width, width))
            img = img.reshape(1, width, width, 3)
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.tensor(img).float()
            imgs[j, ...] = img

        img_name = pair_names[3]
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

    return y, images_vectors_sender, images_vectors_receiver

def get_batch_random_test(opt):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, opt.game_size-1, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    width = 128
    with open('../data/test_pair.txt', 'r') as f:
        pairs = f.readlines()[:32]
    for i in range(opt.batch_size):
        pair = pairs[i][:-1].split()
        target = int(pair[opt.game_size])
        imgs = torch.zeros(opt.game_size-1, 3, 128, 128)
        for j in range(opt.game_size-1):
            if j == target:
                img_name = os.path.join(opt.data_root, pair[opt.game_size-1])
                img = cv2.imread(img_name, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (width, width))
                img = img.reshape(1, width, width, 3)
                img = np.transpose(img, (0, 3, 1, 2))
                img = torch.tensor(img).float()
                img_s = img

            img_name = os.path.join(opt.data_root, pair[j].split('_ske.jpg')[0] + '_img.jpg')
            # print(img_name)
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (width, width))
            img = img.reshape(1, width, width, 3)
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.tensor(img).float()
            imgs[j, ...] = img

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

    return y, images_vectors_sender, images_vectors_receiver

def get_test_batch(opt, loader, i):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, 4, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    batch_inds = range(i, min(i+opt.batch_size, len(loader.dataset)))
    # print(batch_inds)
    for iid, ind in enumerate(batch_inds):
        img_s, target, imgs = loader.dataset[ind]
        if opt.cuda:
            img_s = Variable(img_s.cuda())
            imgs = Variable(imgs.cuda())
        else:
            img_s = Variable(img_s)
            imgs = Variable(imgs)
        images_vectors_sender[iid, ...] = img_s
        images_vectors_receiver[iid, ...] = imgs
        y[iid] = target

    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    return y, images_vectors_sender, images_vectors_receiver

def get_val_batch(opt, loader):
    images_vectors_sender = torch.zeros((opt.batch_size, 3, 128, 128)).cuda()
    images_vectors_receiver = torch.zeros((opt.batch_size, 4, 3, 128, 128)).cuda()
    y = torch.zeros(opt.batch_size).long()
    batch_inds = range(opt.batch_size)
    for iid, ind in enumerate(batch_inds):
        img_s, target, imgs = loader.dataset[ind]
        if opt.cuda:
            img_s = Variable(img_s.cuda())
            imgs = Variable(imgs.cuda())
        else:
            img_s = Variable(img_s)
            imgs = Variable(imgs)
        images_vectors_sender[iid, ...] = img_s
        images_vectors_receiver[iid, ...] = imgs
        y[iid] = target

    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    return y, images_vectors_sender, images_vectors_receiver

def create_val_batch(opt, loader):
    val_z = {}
    val_images_indexes_sender = {}
    val_images_indexes_receiver = {}
    n = 0
    i_game=0
    opt.feat_size = loader.dataset.data_tensor.shape[-1]
    print("N data", loader.dataset.data_tensor.shape[0])
    while True:
        ### GET BATCH INDEXES
        C = len(loader.dataset.obj2id.keys()) #number of concepts
        images_indexes_sender = np.zeros((opt.batch_size,opt.game_size))
        images_indexes_receiver = np.zeros((opt.batch_size,opt.game_size))
        for b in range(opt.batch_size):
            if opt.same:
                # randomly sample 1 concepts
                concepts = np.random.choice(C, 1)
                c1 = concepts[0]
                c2 = c1
                ims1 = loader.dataset.obj2id[c1]["ims"]
                ims2 = loader.dataset.obj2id[c2]["ims"]
                assert np.intersect1d(np.array(ims1),
                    np.array(ims2)).shape[0]== len(ims1)
                # randomly sample 2 images from the same concept
                idxs_sender = np.random.choice(ims1,opt.game_size,replace=False)
                images_indexes_sender[b,:] = idxs_sender
                images_indexes_receiver[b,:] = idxs_sender
            else:
                # randomly sample 2 concepts
                concepts = np.random.choice(C, 2, replace = False)
                c1 = concepts[0]
                c2 = concepts[1]

                ims1 = loader.dataset.obj2id[c1]["ims"]
                ims2 = loader.dataset.obj2id[c2]["ims"]
                assert np.intersect1d(np.array(ims1),
                    np.array(ims2)).shape[0] == 0
                # randomly sample 2 images for each concept
                idx1 = np.random.choice(ims1, 2, replace=False)
                idx2 = np.random.choice(ims2, 2, replace=False)
                idxs_sender = np.array([idx1[0], idx2[0]])
                idxs_receiver = np.array([idx1[1], idx2[1]])
                images_indexes_sender[b,:] = idxs_sender
                images_indexes_receiver[b,:] = idxs_receiver

        images_indexes_sender = torch.LongTensor(images_indexes_sender)
        images_indexes_receiver = torch.LongTensor(images_indexes_receiver)

        # SAVE
        val_images_indexes_sender[i_game] = images_indexes_sender.clone()
        val_images_indexes_receiver[i_game] = images_indexes_receiver.clone()

        # GET BATCH Y
        probas = torch.zeros(2).fill_(0.5)
        val_z_game = torch.zeros(opt.batch_size).long()
        for i in range(opt.batch_size):
            z = torch.bernoulli(probas).long()[0]
            val_z_game[i] = 1
        # SAVE
        val_z[i_game] = val_z_game.clone()

        # INCREMENT
        n += val_z_game.size(0)
        i_game += 1
        if n >= opt.val_images_use:
            break
    return val_z, val_images_indexes_sender, val_images_indexes_receiver

def get_batch_fromsubdataset(opt,loader,indexes):

    sub_concepts=np.unique(loader.dataset.labels[indexes])
    all_concepts=np.unique(loader.dataset.labels)
    sub_C=np.where(np.in1d(all_concepts,sub_concepts))[0]

    # DEBUG
    tmp = sub_concepts.tolist()
    for c in sub_concepts:
        n_c = (loader.dataset.labels[indexes] == c).sum()
        if n_c == 1:
            tmp.remove(c)
    tmp = np.array(tmp)
    sub_C=np.where(np.in1d(all_concepts,tmp))[0]
    images_indexes_sender=np.zeros((opt.batch_size,opt.game_size))
    images_indexes_receiver=np.zeros((opt.batch_size,opt.game_size))
    batch_c=np.zeros((opt.batch_size,opt.game_size),dtype='int')
    for b in range(opt.batch_size):
        if opt.same:
            # NOISE SHOULD ALWAYS BE 0 since concepts are the same!
            assert opt.noise == 0
            # randomly sample 1 concepts
            concepts = np.random.choice(sub_C, 1)
            c1 = concepts[0]
            c2 = c1
            intersect=np.intersect1d(loader.dataset.obj2id[c1]["ims"],indexes)
            # randomly sample 2 images from the same concept
            idxs_sender=np.random.choice(intersect,opt.game_size,replace=False)
            images_indexes_sender[b,:] = idxs_sender
            images_indexes_receiver[b,:] = idxs_sender
        else:
            # randomly sample 2 concepts
            concepts = np.random.choice(sub_C,2,replace = False)
            c1 = concepts[0]
            c2 = concepts[1]
            intersect1=np.intersect1d(loader.dataset.obj2id[c1]["ims"],indexes)
            intersect2=np.intersect1d(loader.dataset.obj2id[c2]["ims"],indexes)
            # randomly sample 2 different images for each concept
            idx1 = np.random.choice(intersect1, 2, replace=False)
            idx2 = np.random.choice(intersect2, 2, replace=False)
            idxs_sender = np.array([idx1[0], idx2[0]])
            idxs_receiver = np.array([idx1[1], idx2[1]])
            images_indexes_sender[b,:] = idxs_sender
            images_indexes_receiver[b,:] = idxs_receiver

        batch_c[b,:] = [c1,c2]
    images_indexes_sender = torch.LongTensor(images_indexes_sender)
    images_vectors_sender = []
    for i in range(opt.game_size):
        x, _ = loader.dataset[images_indexes_sender[:,i]]
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_sender.append(x)

    # THOSE WILL BE USED IF WE HAVE NOISE
    images_indexes_receiver = torch.LongTensor(images_indexes_receiver)
    images_vectors_alternative = []
    for i in range(opt.game_size):
        x, _ = loader.dataset[images_indexes_receiver[:,i]]
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_alternative.append(x)

    y = torch.zeros((opt.batch_size,2)).long()
    ### shuffle the images and fill the ground_truth
    # FILL WITH ZEROS
    images_vectors_receiver = []
    for i in range(opt.game_size):
        x = torch.zeros((opt.batch_size,opt.feat_size))
        if opt.cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(x)
        images_vectors_receiver.append(x)

    probas = torch.zeros(2).fill_(0.5)
    # #TODO: make a faster function, here explicit for debugging later
    for i in range(opt.batch_size):
        z = torch.bernoulli(probas).long()[0]
        y[i,z] = 1
        if not opt.noise:
            referent = images_vectors_sender[0][i,:]
            non_referent = images_vectors_sender[1][i,:]
        elif opt.noise: # use alternative images of the same concepts
            referent = images_vectors_alternative[0][i,:]
            non_referent = images_vectors_alternative[1][i,:]
        if z == 0:
            #sets requires_grad to True if needed
            images_vectors_receiver[0][i,:] = referent.clone()
            images_vectors_receiver[1][i,:] = non_referent.clone()
        elif z == 1:
            #sets requires_grad to True if needed
            images_vectors_receiver[0][i,:] = non_referent.clone()
            images_vectors_receiver[1][i,:] = referent.clone()
    if opt.cuda:
        y = Variable(y.cuda())
    else:
        y = Variable(y)

    # compute a new value, the inputs similarity, to be used for the sender
    sims_im_s = torch.zeros(opt.batch_size)
    sims_im_s=sims_im_s.cuda()
    for b in range(opt.batch_size):
        im1 = images_vectors_sender[0][b,:].data.unsqueeze(0)
        im2 = images_vectors_sender[1][b,:].data.unsqueeze(0)
        space = torch.cat([im1, im2],dim=0)
        sims_im_s[b]=compute_similarity_images(space)
    sims_im_s=Variable(sims_im_s)
    # compute a new value, the inputs similarity, to be used for the receiver
    sims_im_r = torch.zeros(opt.batch_size)
    sims_im_r=sims_im_r.cuda()
    for b in range(opt.batch_size):
        im1 = images_vectors_receiver[0][b,:].data.unsqueeze(0)
        im2 = images_vectors_receiver[1][b,:].data.unsqueeze(0)
        space = torch.cat([im1, im2],dim=0)
        sims_im_r[b]=compute_similarity_images(space)
    sims_im_r=Variable(sims_im_r)
    return x, y, images_vectors_sender,images_indexes_sender, \
            images_vectors_receiver,images_indexes_receiver,batch_c,sims_im_s, sims_im_r

def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
  def __init__(self, modules: Iterable[Module]):
      """
      Context manager to locally freeze gradients.
      In some cases with can speed up computation because gradients aren't calculated for these listed modules.
      example:
      ```
      with FreezeParameters([module]):
          output_tensor = module(input_tensor)
      ```
      :param modules: iterable of modules. used to call .parameters() to freeze gradients.
      """
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          param.requires_grad = False

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]
