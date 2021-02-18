import torch
import torch.nn as nn
import os


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing + cross entropy by PistonY (Thanks!)
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_inception_v3(num_classes=1000, pretrained=True):
    """
    get inception v3 from torch.hub and freeze all layer except FC layers
    this is a work for transfer learning
    :param num_classes: number of dataset classes that you want to train, default 1000
    :param pretrained: parameters pretrained option, default True
    :return: model for transfer learning
    """
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=pretrained)
    for param in model.parameters():
        param.require_grads=False

    aux_in = model.AuxLogits.fc.in_features
    fc_in = model.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
    model.fc = nn.Linear(fc_in, num_classes)
    return model


def save_model(model, model_name):
    """
    save model in './saved_model/model_name.pt' path
    :param model: model you want to save
    :param model_name: this will be the name of the model file
    """
    path = './saved_model'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, path + '/' + str(model_name) + '.pt')


def load_model(model_name):
    """
    load model in './saved_model/model_name.pt' path
    :param model_name: model name you want to load
    """
    path = './saved_model/' + str(model_name) + '.pt'
    if not os.path.exists(path):
        print('No such model exist!')
        return None
    else:
        model = torch.load(path)
        return model


if __name__ == '__main__':
    model = get_inception_v3(num_classes=200)
    print(model)
    save_model(model, 'test')

