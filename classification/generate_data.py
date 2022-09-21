import pickle
from utils import class2ID
import cv2
import numpy as np
import os

cates = {}
sketches = {}
save_path = 'data/cate97_gt_sketch/'
data_path = '../signal_game/to_cluster_comp_rs0'
with open('{}/img_name.p'.format(data_path), 'rb') as f:
    img_names = pickle.load(f)
with open('{}/sketch.p'.format(data_path), 'rb') as f:
    imgs = pickle.load(f)
data_root = '../../synthesizing_human_like_sketches/output2/'
width = 128

cnt = 0
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
    img_name = img_name.split('_ske')[0] + '_gt.jpg'
    print(data_root + img_name)
    img = cv2.imread(data_root + img_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    sketches[cate].append(img)
    # sketches[cate].append(imgs[iid])

train_img_names = []
train_y = []
test_img_names = []
test_y = []
train_imgs = []
test_imgs = []

for cate in cates.keys():
    train_img_names += cates[cate][:7]
    test_img_names += cates[cate][7:]
    train_y += [class2ID[cate]]*7
    test_y += [class2ID[cate]]*len(cates[cate][7:])
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

