import torch
import torch.nn as nn
import torch.nn.functional as F


class CCE(nn.Module):
    def __init__(self, device, balancing_factor=1):
        super(CCE, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.device = device # {'cpu', 'cuda:0', 'cuda:1', ...}
        self.balancing_factor = balancing_factor

    def forward(self, yHat, y):
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # complement entropy
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        Px = yHat / (1 - Yg) + 1e-7
        Px_log = torch.log(Px + 1e-10)
        y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.to(device=self.device)
        complement_entropy = torch.sum(output) / (float(batch_size) * float(yHat.shape[1]))

        return cross_entropy - self.balancing_factor * complement_entropy



class ESPL_CE(nn.Module):
    def __init__(self):
        super(ESPL_CE, self).__init__()
        # self.nll_loss = nn.NLLLoss()
        # self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', size_average = True)
        # self.softmax = nn.Softmax2d()
    def forward(self, yHat, y, confidence):
        # # cross entropy
        # cross_entropy1 = self.ce_loss(yHat, y)
        # # cross entropy
        # cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # hand cross entropy
        P_i = torch.nn.functional.log_softmax(yHat, dim=1)
        y = torch.nn.functional.one_hot(y,num_classes=6)
        y = y.permute(0, 3, 1, 2)
        loss = y*(P_i + 0.000000000001)
        loss = torch.sum(loss, dim=1)
        # confidence = (confidence - torch.min(confidence)) /(torch.max(confidence)-torch.min(confidence))
        # loss = loss * (1-confidence)
        loss = torch.mean(loss, dim=(0,1,2))
        hand_cross_entropy = -1*loss
        # print("--------------------------------------")
        # print("cross_entropy1:"+str(cross_entropy1.item()))
        # print("cross_entropy:"+str(cross_entropy.item()))
        # print("hand_cross_entropy:"+str(hand_cross_entropy.item()))
        # print("--------------------------------------")
        return hand_cross_entropy