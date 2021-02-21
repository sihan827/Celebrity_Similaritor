import torch
import model
import torch.nn as nn
import utils
import dataset
from torch.optim.lr_scheduler import StepLR
import test
import os
import pickle


def train(train_model, train_loader, valid_loader, meters, hyperparam_dict):
    """
    function to train model and calculate logs
    if 'system' is true, it means classification model, else it means regression model
    you can change this function if you want(ex: change optimizer, criterion)
    :param train_model: model you want to train
    :param train_loader: DataLoader for train set
    :param valid_loader: DataLoader for valid set
    :param meters: AverageMeter instances List used for calculating logs, must be size 3 list
    :param hyperparam_dict: criterion used for calculating loss and doing back propagation
    :return: train logs list, test logs list
    """
    USE_CUDA = torch.cuda.is_available()
    DEVICE = 'cuda' if USE_CUDA else 'cpu'
    system = True if train_model.fc.out_features != 1 else False
    checkpoint_path = './checkpoint.pt'
    if system:
        train_result = {
            'train_loss_list': [],
            'train_top1_list': [],
            'train_top5_list': []
        }
        test_result = {
            'test_loss_list': [],
            'test_top1_list': [],
            'test_top5_list': []
        }
    else:
        train_result = {
            'train_loss_list': []
        }
        test_result = {
            'test_loss_list': []
        }
    loss_check = 0
    epoch = hyperparam_dict['epoch']
    lr = hyperparam_dict['lr']
    if system:
        smoothing = hyperparam_dict['smoothing']
        num_classes = hyperparam_dict['num_classes']

    if system:
        criterion = model.LabelSmoothingLoss(num_classes, smoothing)
    else:
        criterion = nn.L1Loss()

    train_model = train_model.to(DEVICE)
    if system:
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(train_model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=2, gamma=0.94)

    for iter in range(epoch):
        train_model.train()
        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            N = X.shape[0]

            optimizer.zero_grad()
            outs = train_model(X)
            output = outs.logits
            aux = outs.aux_logits

            output_loss = criterion(output, y)
            aux_loss = criterion(aux, y)
            loss = output_loss + aux_loss * 0.3

            loss.backward()
            optimizer.step()

            if system:
                prec1, prec3 = utils.accuracy(output, y, topk=(1, 5))
                meters[0].update(prec1.item(), N)
                meters[1].update(prec3.item(), N)
            meters[2].update(loss.item(), N)

            if step % 100 == 0:
                print('[%d %5d] loss : %f' % (iter, step, loss.item()))
        scheduler.step()

        if system:
            top1_avg = meters[0].get_avg()
            top5_avg = meters[1].get_avg()
        loss_avg = meters[2].get_avg()

        for avg in [meters[0], meters[1], meters[2]]:
            avg.reset()

        if system:
            print('Epoch[{cur_epoch}/{max_epoch}] [Train] Loss: {loss:.4f}, Top1: {top1: .4f}, Top5: {top5: .4f}'.format(
                cur_epoch=iter,
                max_epoch=epoch,
                loss=loss_avg,
                top1=top1_avg,
                top5=top5_avg
            ))
        else:
            print('Epoch[{cur_epoch}/{max_epoch}] [Train] Loss: {loss:.4f}'.format(
                cur_epoch=iter,
                max_epoch=epoch,
                loss=loss_avg
            ))

        if loss_check > loss_avg:
            torch.save({
                'epoch': iter,
                'model_state_dict': train_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_avg
            }, checkpoint_path)
        loss_check = loss_avg

        train_result['train_loss_list'].append(loss_avg)
        if system:
            train_result['train_top1_list'].append(top1_avg)
            train_result['train_top5_list'].append(top5_avg)

        if system:
            test_loss, test_top1, test_top5 = test.test(train_model, valid_loader, meters, criterion)
        else:
            test_loss = test.test(train_model, valid_loader, meters, criterion)

        test_result['test_loss_list'].append(test_loss)
        if system:
            test_result['test_top1_list'].append(test_top1)
            test_result['test_top5_list'].append(test_top5)

    return train_result, test_result


def celebrities_face_train(name, deploy=False):
    """
    get celebrities_face dataset and train model by dataset
    after train, it saves model and logs file
    :param name: name of model, used for file name
    :param deploy: if True, train uses train set and test set, else train uses train set and valid set, default False
    """
    batch_size = 32
    epoch = 100
    lr = 0.001
    smoothing = 0.1
    dset = dataset.get_dataset(
        './celebrities_face',
        [0.5893, 0.4750, 0.4330],
        [0.2573, 0.2273, 0.2134],
    )

    num_classes = len(dset.classes)

    hyperparam_dict = {
        'epoch': epoch,
        'lr': lr,
        'smoothing': smoothing,
        'num_classes': num_classes
    }

    meters = [
        utils.AverageMeter(),
        utils.AverageMeter(),
        utils.AverageMeter()
    ]

    train_set, test_set, train_labels = dataset.train_test_set_split(dset, 'celebrities_face')
    if deploy:
        train_loader, valid_loader = dataset.train_valid_loader_split(train_set, train_labels, batch_size, valid_size=0.1)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        valid_loader = torch.utils.data.DataLoader(
            test_set, batch_size=32, num_workers=2
        )

    train_model = model.get_inception_v3(num_classes=num_classes)
    result_log = train(train_model, train_loader, valid_loader, meters, hyperparam_dict)

    model.save_model(train_model, name)
    save_log(result_log, name)


def face_age_train(name, deploy=False):
    """
    get face_age dataset and train model by dataset
    after train, it saves model and logs file
    :param name: name of model, used for file name
    :param deploy: if True, train uses train set and test set, else train uses train set and valid set, default False
    """
    batch_size = 32
    epoch = 150
    lr = 0.01
    dset = dataset.get_dataset(
        './face_age',
        [0.6361, 0.4875, 0.4189],
        [0.2105, 0.1893, 0.1820],
        target_transform=dataset.face_target
    )

    hyperparam_dict = {
        'epoch': epoch,
        'lr': lr,
    }

    meters = [
        utils.AverageMeter(),
        utils.AverageMeter(),
        utils.AverageMeter()
    ]

    train_set, test_set, train_labels = dataset.train_test_set_split(dset, 'face_age', test_size=0.2)
    if deploy:
        train_loader, valid_loader = dataset.train_valid_loader_split(train_set, train_labels, batch_size, valid_size=0.5)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        valid_loader = torch.utils.data.DataLoader(
            test_set, batch_size=32, num_workers=2
        )

    train_model = model.get_inception_v3(num_classes=1)
    result_log = train(train_model, train_loader, valid_loader, meters, hyperparam_dict)

    model.save_model(train_model, name)
    save_log(result_log, name)


def save_log(result_seq, test_name):
    """
    save logs in logs list
    :param result_seq: logs list
    :param test_name: name of test
    """
    path = './log'
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/' + str(test_name) + '.pickle'
    with open(path, 'wb') as fw:
        pickle.dump(result_seq, fw)


def load_log(test_name):
    """
    load logs from file
    :param test_name: name of test
    """
    path = './log/' + str(test_name) + '.pickle'
    if not os.path.exists(path):
        print('No such log file exists!')
        return None
    else:
        with open(path, 'rb') as fr:
            loaded_log = pickle.load(fr)
        return loaded_log


if __name__ == '__main__':
    face_age_train('test199')