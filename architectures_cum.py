
from reinforce_one_stroke import *
from resnet import ResNet
import numpy as np
import torchvision.models as models
import pdb

def cal_reward(one_hot, probs, gt, opt, cnt, mask):
    _, amax = one_hot.max(dim=1)
    reward = torch.where(amax < opt.game_size-1, ((amax == gt).float()*2-1), torch.tensor(-opt.step_cost).float().cuda())

    log_receiver_probs = torch.log(probs)
    if np.any(np.isnan(log_receiver_probs.data.clone().cpu().numpy())):
        pdb.set_trace()
    masked_log_proba_receiver = (one_hot * log_receiver_probs).sum(1)
    rewards_no_grad = reward.detach()
    a = - ((rewards_no_grad)
                        * masked_log_proba_receiver * mask)
    if np.any(np.isnan(a.data.clone().cpu().numpy())):
        pdb.set_trace()
    return - (rewards_no_grad)*mask, masked_log_proba_receiver, rewards_no_grad

class Players(nn.Module):
    def __init__(self, sender, receiver):
        super(Players, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = 128

    def forward(self, images_vectors, images_vectors_receiver,
                opt, gt, train=True):
        cnt = torch.zeros(opt.batch_size).cuda()
        res = np.zeros((opt.batch_size, opt.max_step, 3, 128, 128))

        canvas = torch.zeros([opt.batch_size, 3, self.width, self.width], requires_grad=False).to(self.device)
        total_loss = []
        total_prob = []
        total_reward = torch.zeros(opt.batch_size).cuda()
        precision = np.zeros(opt.batch_size)
        mask = torch.ones(opt.batch_size).long().cuda()
        choice = -1*np.ones(opt.batch_size)
        step_num = 0

        for i in range(opt.max_step):
            canvas0 = canvas.detach()
            canvas = sender_action_grad(self.sender,
                                   images_vectors, i, canvas0, opt.num_stroke)

            one_hot_output, receiver_probs = receiver_action_two_step(self.receiver,
                                                                      images_vectors_receiver, canvas0, canvas, opt,
                                                                      step_num, train)
            _, amax = one_hot_output.max(dim=1)
            cnt = cnt + 1 * mask
            step_loss, step_prob, step_reward = cal_reward(one_hot_output, receiver_probs, gt, opt, i, mask)
            total_loss.append(step_loss)
            total_prob.append(step_prob)
            total_reward += step_reward
            next_mask = mask * (amax == opt.game_size-1)
            if not train:
                mask_ = mask.cpu().numpy().astype(np.bool)
                next_mask_ = next_mask.cpu().numpy().astype(np.bool)
                res[mask_, i] = canvas.cpu().numpy()[mask_]
                choice[mask_ * ~next_mask_] = amax.cpu().numpy()[mask_ * ~next_mask_]
                precision[mask_ * ~next_mask_] += (gt == amax).float().cpu().numpy()[mask_ * ~next_mask_]
            if mask.sum() == 0:
                break
            mask = next_mask
            step_num += 1

        running_add = torch.zeros(opt.batch_size, requires_grad=False).cuda()
        discounted_accumulated_rewards = torch.zeros((len(total_loss), opt.batch_size), requires_grad=False).cuda()
        total_prob = torch.stack(total_prob)
        for i in reversed(range(len(total_loss))):
            running_add = running_add*opt.discount + total_loss[i]
            discounted_accumulated_rewards[i] = running_add
        final_loss = discounted_accumulated_rewards*total_prob
        final_loss = final_loss.sum(dim=0)
        return final_loss, res, precision, cnt.cpu().numpy().astype(int), choice, total_reward

class Sender(nn.Module):
    def __init__(self, device, max_step, opt, width = 128):
        super(Sender, self).__init__()
        self.feature = ResNet(9, 18, 45) # 5*9(6+3)
        self.feature.load_state_dict(torch.load(opt.sender_path))

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
        actions = self.feature(torch.cat([canvas, x/255, stepnum, self.coord], 1))
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
        stepnum = self.T * step / self.max_step
        out = self.feature(torch.cat([canvas, x/255, stepnum], 1))
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
        features = self.feature(x/255)
        features = self.lin1(features)
        features = features.view(-1, self.game_size-1, 100)
        can_feature0 = self.feature(canvas0)
        can_feature1 = self.feature(canvas1)
        can_feature0 = self.lin1(can_feature0).view(-1, 1, 100)
        features = torch.cat([features, can_feature0], dim=1)

        can_feature = self.lin2(can_feature1)

        can_feature = can_feature.view(-1, 100, 1)

        out = torch.bmm(features, can_feature)
        out = out.squeeze(dim=-1)

        probas = F.softmax(out, dim=1)

        return probas

