import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = (nn.Linear(10, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)

Decoder = FCN()
Decoder.load_state_dict(torch.load('./pretrained/renderer.pkl'))
Decoder = Decoder.to(device).eval()

def variable_hook(grad):
    return grad

def one_hot(y,depth,cuda=True):
    if not cuda:
        y_onehot = torch.FloatTensor(y.size(0),depth)
    else:
        y_onehot = torch.cuda.FloatTensor(y.size(0),depth)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.data.unsqueeze(1), 1)
    return Variable(y_onehot)

def decode_con(x, canvas, num_stroke):
    with torch.no_grad():
        x = x.view(-1, 6 + 3)
        scale = torch.tensor([0, 0, 1, 1])
        scale = torch.stack([scale] * x.shape[0]).cuda()
        x_ = torch.cat([x[:, :6], scale], dim=1)
        stroke = 1 - Decoder(x_)
        stroke = stroke.view(-1, width, width, 1)
        color = torch.ones(x.shape[0], 3).cuda()
        color_stroke = stroke * color.view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        stroke = stroke.view(-1, num_stroke, 1, width, width)
        color_stroke = color_stroke.view(-1, num_stroke, 3, width, width)
        res = []
        x_his = []
        for i in range(num_stroke):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
            res.append(stroke[:, i])
            x_his.append(x[i, :6])
    return canvas, res, x_his

def sender_action_noise(sender, image, step, canvas, train):
    actions = sender(image, step, canvas)
    if train:
        actions += torch.normal(0, 1, actions.shape)
    canvas = decode_con_grad(actions, canvas)
    return canvas

def sender_action_grad(sender, image, step, canvas, add_noise=False, num_stroke=3):
    actions = sender(image, step, canvas)
    if add_noise:
        noise = torch.normal(0, 0.05, size=actions.shape).cuda()
        actions_ = actions + noise
        canvas = decode_con_grad(actions_, canvas, num_stroke)
    else:
        canvas = decode_con_grad(actions, canvas, num_stroke)
    return canvas

def decode_con_grad(x, canvas, num_stroke):
    x = x.view(-1, 6 + 3)
    scale = torch.tensor([0, 0, 1, 1])
    scale = torch.stack([scale] * x.shape[0]).cuda()
    x_ = torch.cat([x[:, :6], scale], dim=1)
    stroke = 1 - Decoder(x_)
    stroke = stroke.view(-1, width, width, 1)
    color = torch.ones(x.shape[0], 3).cuda()
    color_stroke = stroke * color.view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, num_stroke, 1, width, width)
    color_stroke = color_stroke.view(-1, num_stroke, 3, width, width)
    for i in range(num_stroke):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas

def sender_action(sender, image, step, canvas, add_noise=False, num_stroke=3):
    actions = sender(image, step, canvas)
    if add_noise:
        noise = torch.normal(0, 0.05, size=actions.shape).cuda()
        actions_ = actions + noise
        canvas, _, _ = decode_con(actions_, canvas, num_stroke)
    else:
        canvas, _, _ = decode_con(actions, canvas, num_stroke)
    
    return canvas

def receiver_action(receiver, images_vectors, sender_canvas0, canvas1, opt, cnt):
    receiver_probs = receiver(images_vectors, sender_canvas0, canvas1, cnt)
    receiver_probs = receiver_probs + receiver.eps
    sample = torch.multinomial(receiver_probs, 1)
    sample = sample.squeeze(-1)
    one_hot_output = one_hot(sample, receiver.game_size, cuda=opt.cuda)
    one_hot_output = Variable(one_hot_output.data, requires_grad = True)
    return one_hot_output, receiver_probs

def receiver_action_two_step(receiver, images_vectors, canvas0, canvas1, opt, cnt, is_train=True):
    receiver_probs = receiver(images_vectors, canvas0, canvas1)
    receiver_probs = receiver_probs + receiver.eps
    if is_train:
        sample = torch.multinomial(receiver_probs, 1)
        sample = sample.squeeze(-1)
    else:
        _, sample = receiver_probs.max(dim=1)

    one_hot_output = one_hot(sample, receiver.game_size, cuda=opt.cuda)
    one_hot_output = Variable(one_hot_output.data, requires_grad = True)
    return one_hot_output, receiver_probs

def receiver_action_retrieve(receiver, images_vectors, canvas0, canvas1, opt, is_train=True):
    receiver_probs = receiver(images_vectors, canvas0, canvas1)
    receiver_probs = receiver_probs + receiver.eps
    if is_train:
        sample = torch.multinomial(receiver_probs, 1)
        sample = sample.squeeze(-1)
    else:
        _, sample = receiver_probs.max(dim=1)

    one_hot_output = one_hot(sample, receiver.game_size, cuda=opt.cuda)
    one_hot_output = Variable(one_hot_output.data, requires_grad = True)
    return one_hot_output, receiver_probs

class Communication(torch.nn.Module):
    def __init__(self):
        super(Communication, self).__init__()

    def forward(self, y, predictions, cnt):

        _, amax = predictions.max(dim=1)
        _, amax_gt = y.max(dim=1)
        rewards = (amax == amax_gt).float()/cnt

        return rewards
