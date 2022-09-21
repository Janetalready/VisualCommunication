import torch
import torch.nn as nn
import torch.nn.functional as F
from reinforce_one_stroke import *
from resnet import ResNet
import numpy as np
import torchvision.models as models
import pdb
import copy
from utils_dreamer import *


def cal_reward(one_hot, probs, gt, opt, cnt, mask):
    _, amax = one_hot.max(dim=1)
    reward = torch.where(amax < opt.game_size - 1, ((amax == gt).float() * 2 - 1),
                         torch.tensor(-opt.step_cost).float().cuda())
    rewards_no_grad = reward.detach()
    return (rewards_no_grad) * mask, probs


class Players(nn.Module):
    def __init__(self, sender, receiver, value_model):
        super(Players, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.value_model = value_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = 128

    def forward(self, images_vectors, images_vectors_receiver,
                opt, gt, train=True):
        cnt = torch.zeros(opt.batch_size).cuda()
        res = np.zeros((opt.batch_size, opt.max_step, 3, 128, 128))

        canvas = torch.zeros([opt.batch_size, 3, self.width, self.width], requires_grad=False).to(self.device)
        total_reward = []
        total_prob = []
        total_value = []
        total_canvas0 = []
        total_canvas1 = []
        total_mask = []
        precision = np.zeros(opt.batch_size)
        actual_reward = torch.zeros(opt.batch_size).cuda()
        mask = torch.ones(opt.batch_size).long().cuda()
        choice = -1 * np.ones(opt.batch_size)
        step_num = 0
        flag = 1
        for i in range(opt.max_step+1):
            canvas0 = canvas.detach()
            if not opt.sender_fixed:
                canvas = sender_action_grad(self.sender,
                                            images_vectors, i, canvas0, opt.sender_add_noise, opt.num_stroke)
            else:
                canvas = sender_action(self.sender,
                                            images_vectors, i, canvas0, opt.sender_add_noise, opt.num_stroke)

            one_hot_output, receiver_probs = receiver_action_two_step(self.receiver,
                                                                      images_vectors_receiver, canvas0, canvas, opt,
                                                                      step_num, train)
            if i < (opt.max_step-1):
                one_hot = torch.FloatTensor([0]*(opt.game_size-1)+[1])
                one_hot_output = torch.stack([one_hot]*opt.batch_size).cuda()
                receiver_probs = torch.stack([one_hot]*opt.batch_size).cuda()
            _, amax = one_hot_output.max(dim=1)

            step_reward, step_prob = cal_reward(one_hot_output, receiver_probs, gt, opt, i, mask)
            actual_reward += step_reward

            with FreezeParameters(self.value_model.modules):
                value = self.value_model(images_vectors, images_vectors_receiver, canvas0, canvas).view(-1, 1)
            total_value.append(value)  # 8, 1
            if flag:
                cnt = cnt + 1 * mask
                total_reward.append(step_reward.view(-1, 1))  # 8, 1
                total_prob.append(step_prob)  # 8, 4
                total_canvas0.append(canvas0)  # 8, 3, 128, 128
                total_canvas1.append(canvas.detach())
                total_mask.append(mask)
            next_mask = mask * (amax == opt.game_size - 1)
            if (not train) and flag:
                mask_ = mask.cpu().numpy().astype(np.bool)
                next_mask_ = next_mask.cpu().numpy().astype(np.bool)
                res[mask_, i] = canvas.cpu().numpy()[mask_]
                choice[mask_ * ~next_mask_] = amax.cpu().numpy()[mask_ * ~next_mask_]
                precision[mask_ * ~next_mask_] += (gt == amax).float().cpu().numpy()[mask_ * ~next_mask_]
            if mask.sum() == 0:
                break
            if next_mask.sum() == 0 or i == (opt.max_step-1):
                flag = 0
            mask = next_mask
            step_num += 1


        value_pred = torch.cat(total_value[:-1], 1)  # 8, 3
        imged_reward = torch.cat(total_reward, 1)  # 8, 3
        next_values = torch.cat(total_value[1:], 1)  # 8, 3
        bootstrap = next_values[:, -1]  # 8, 1
        discount = opt.discount
        discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
        last = bootstrap
        indices = reversed(range(len(total_mask)))
        outputs = []
        next_mask = total_mask[1:] + [torch.zeros(opt.batch_size).cuda()]
        for index in indices:
            disc = discount_tensor[:, index]
            inp = torch.where(next_mask[index] > 0,
                              imged_reward[:, index] + disc * next_values[:, index] * (1 - opt.lambda_),
                              imged_reward[:, index])
            last = torch.where(total_mask[index] > 0, inp + disc * opt.lambda_ * last, value_pred[:, index])
            last = torch.where((next_mask[index] + total_mask[index]) == 1, imged_reward[:, index], last)
            outputs.append(last)
        outputs = list(reversed(outputs))
        outputs = torch.stack(outputs, 1)
        returns = outputs

        total_loss = []
        for i in range(len(total_prob)):
            probs = total_prob[i]  # B*4
            reward = torch.tensor([[-1.] * (opt.game_size - 1) + [0.]] * len(probs), requires_grad=False).cuda()  # B*4
            reward[torch.arange(0, len(probs)), gt] = 1
            if i == (len(total_prob) - 1):
                final_reward = torch.where(reward == 0, reward + opt.discount*next_values[:, i].view(-1, 1), reward)
            else:
                final_reward = torch.where(reward == 0, reward + opt.discount*returns[:, i+1].view(-1, 1), reward)
            total_loss.append((probs * final_reward).sum(dim=1).view(-1, 1))

        return torch.cat(total_loss, 1), res, precision, cnt.cpu().numpy().astype(
            int), choice, actual_reward, total_canvas0, total_canvas1, returns, value_pred


class Sender(nn.Module):
    def __init__(self, device, max_step, opt, width=128):
        super(Sender, self).__init__()
        self.feature = ResNet(9, 18, 9*opt.num_stroke)  # 5*9(6+3)
        self.feature.load_state_dict(torch.load(opt.sender_path))
        if opt.sender_fix_resnet:
            ct = 0
            for child in self.feature.children():
                ct += 1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False

        self.width = width
        self.device = device
        self.max_step = max_step

        coord = torch.zeros([1, 2, width, width])
        for i in range(width):
            for j in range(width):
                coord[0, 0, i, j] = i / (width - 1.)
                coord[0, 1, i, j] = j / (width - 1.)
        coord = coord.to(device)
        self.coord = coord.expand(opt.batch_size, 2, 128, 128)
        self.T = torch.ones([opt.batch_size, 1, self.width, self.width], dtype=torch.float32).to(device)

    def forward(self, x, step, canvas):
        stepnum = self.T * step / self.max_step
        actions = self.feature(torch.cat([canvas, x / 255, stepnum, self.coord], 1))
        return actions


class Receiver(nn.Module):
    def __init__(self, device, game_size, width, opt, eps=1e-8):
        super(Receiver, self).__init__()
        self.feature = ResNet(16, 18, 5)
        self.eps = eps
        self.game_size = game_size
        self.width = width
        self.max_step = opt.max_step
        self.T = torch.ones([opt.batch_size, 1, self.width, self.width], dtype=torch.float32).to(device)

    def forward(self, x, canvas, step):
        x = x.view(canvas.shape[0], -1, x.shape[-1], x.shape[-1])
        # for i in range(0, 12, 3):
        #     a = x[0, i:i+3, ...].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        #     cv2.imshow('img_{}'.format(i), a)
        #     cv2.waitKey(2000)
        stepnum = self.T * step / self.max_step
        out = self.feature(torch.cat([canvas, x / 255, stepnum], 1))
        probas = F.softmax(out, dim=1)
        return probas


class ReceiverOnestep(nn.Module):
    def __init__(self, device, game_size, width, opt, eps=1e-8):
        super(ReceiverOnestep, self).__init__()
        self.feature = models.vgg16(pretrained=True)
        self.feature.classifier = self.feature.classifier[:-1]
        for child in self.feature.children():
            # ct += 1
            # if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

        self.lin1 = nn.Linear(4096, 100, bias=False)
        self.lin2 = nn.Linear(4096, 100, bias=False)

        self.eps = eps
        self.game_size = game_size
        self.width = width
        self.max_step = opt.max_step
        self.T = torch.ones([opt.batch_size, 1, self.width, self.width], dtype=torch.float32).to(device)

    def forward(self, x, canvas0, canvas1):
        x = x.view(-1, 3, self.width, self.width)
        features = self.feature(x / 255)
        features = self.lin1(features)
        features = features.view(-1, self.game_size - 1, 100)
        can_feature0 = self.feature(canvas0)
        can_feature1 = self.feature(canvas1)
        can_feature0 = self.lin1(can_feature0).view(-1, 1, 100)
        features = torch.cat([features, can_feature0], dim=1)

        # a = canvas[0, ...]*255
        # a = a.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        # cv2.imshow('img_{}'.format(1), a)
        # cv2.waitKey(2000)
        can_feature = self.lin2(can_feature1)

        can_feature = can_feature.view(-1, 100, 1)

        out = torch.bmm(features, can_feature)
        out = out.squeeze(dim=-1)

        probas = F.softmax(out, dim=1)

        return probas


class ValueModelTemp(nn.Module):
    def __init__(self, width, game_size):
        super(ValueModelTemp, self).__init__()
        self.feature = models.vgg16(pretrained=True)
        self.feature.classifier = self.feature.classifier[:-1]
        for child in self.feature.children():
            # ct += 1
            # if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

        self.lin1 = nn.Linear(4096, 100, bias=False)
        self.lin2 = nn.Linear(4096, 100, bias=False)
        self.lin3 = nn.Linear(4096, 100, bias=False)
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 1)
        self.act_fn = nn.ELU()
        self.modules = [self.lin1, self.lin2, self.lin3, self.fc1, self.fc2]
        self.width = width
        self.game_size = game_size

    def forward(self, receiver, canvas0, canvas1):
        # sender = sender.view(-1, 3, self.width, self.width)
        receiver = receiver.view(-1, 3, self.width, self.width)  # 8, 3, 3, 128, 128
        features = self.feature(receiver / 255)  # 24, 4096
        features = self.lin1(features)
        features = features.view(-1, self.game_size - 1, 100)
        features = features.sum(dim=1).view(-1, 1, 100)

        # sender_feature = self.feature(sender/255)
        # sender_feature = self.lin1(sender_feature).view(-1, 1, 100)

        can_feature0 = self.feature(canvas0)  # 8, 3, 128, 128
        can_feature1 = self.feature(canvas1)
        can_feature0 = self.lin2(can_feature0).view(-1, 1, 100)
        can_feature = self.lin3(can_feature1).view(-1, 1, 100)  # 8, 1, 100
        features = torch.cat([features, can_feature0, can_feature], dim=1).view(can_feature.shape[0], -1)  # 8, 300

        hidden = self.act_fn(self.fc1(features))
        value = self.fc2(hidden)
        return value  # 8, 1


class ValueModel(nn.Module):
    def __init__(self, width, game_size):
        super(ValueModel, self).__init__()
        self.feature = models.vgg16(pretrained=True)
        self.feature.classifier = self.feature.classifier[:-1]
        for child in self.feature.children():
            # ct += 1
            # if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

        self.lin1 = nn.Linear(4096, 100, bias=False)
        self.lin2 = nn.Linear(4096, 100, bias=False)
        self.lin3 = nn.Linear(4096, 100, bias=False)
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, 1)
        self.act_fn = nn.ELU()
        self.modules = [self.lin1, self.lin2, self.lin3, self.fc1, self.fc2]
        self.width = width
        self.game_size = game_size

    def forward(self, sender, receiver, canvas0, canvas1):
        sender = sender.view(-1, 3, self.width, self.width)
        receiver = receiver.view(-1, 3, self.width, self.width)  # 8, 3, 3, 128, 128
        features = self.feature(receiver / 255)  # 24, 4096
        features = self.lin1(features)
        features = features.view(-1, self.game_size - 1, 100)
        features = features.sum(dim=1).view(-1, 1, 100)

        sender_feature = self.feature(sender / 255)
        sender_feature = self.lin1(sender_feature).view(-1, 1, 100)

        can_feature0 = self.feature(canvas0)  # 8, 3, 128, 128
        can_feature1 = self.feature(canvas1)
        can_feature0 = self.lin2(can_feature0).view(-1, 1, 100)
        can_feature = self.lin3(can_feature1).view(-1, 1, 100)  # 8, 1, 100
        features = torch.cat([features, sender_feature, can_feature0, can_feature], dim=1).view(can_feature.shape[0],
                                                                                                -1)  # 8, 300

        hidden = self.act_fn(self.fc1(features))
        value = self.fc2(hidden)
        return value  # 8, 1