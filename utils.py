import torch
import torch.nn.functional as F

class AverageMeter(object):
    """
    class for calculating log values while training
    """
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    only works for classification models
    :param output: output of batch forwarding
    :param target: real label target value
    :param topk: types of accuracy rate you want to calculate
    :return: accuracy list
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def get_topk_prob(output, classes, topk=5):
    """
    get real probabilities for topk classes
    :param output: output of batch forwarding
    :param classes: DataSet classes
    :param topk: type of probabilities you want to calculate
    :return: topk probabilities dictionary
    """
    probs = F.softmax(output[0], dim=0)
    topk_probs, topk_id = torch.topk(probs, topk)
    probs_dict = {}
    for i in range(topk_probs.size(0)):
        probs_dict[classes[topk_id][i]] = topk_probs[i].item()
    return probs_dict