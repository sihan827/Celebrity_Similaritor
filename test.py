import torch
import model
import dataset
import utils


def test(test_model, loader, meters, criterion):
    """
    function to test model and calculate logs
    calculate average top1, top5 accuracy and loss by criterion in classification model
    calculate average loss by criterion in regression model
    :param model: trained model to test
    :param loader: DataLoader loaded from test DataSet
    :param meters: AverageMeter instances List used for calculating logs, must be size 3 list
    :param criterion: criterion used for calculating loss
    :return:
    """
    USE_CUDA = torch.cuda.is_available()
    DEVICE = 'cuda' if USE_CUDA else 'cpu'

    test_model.to(DEVICE)
    test_model.eval()
    system = True if test_model.fc.out_features != 1 else False
    with torch.no_grad():
        for step, (X, y) in enumerate(loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            N = X.shape[0]

            output = test_model(X)
            loss = criterion(output, y)
            if system:
                prec1, prec3 = utils.accuracy(output, y, topk=(1, 5))
                meters[0].update(prec1.item(), N)
                meters[1].update(prec3.item(), N)
            meters[2].update(loss.item(), N)

        if system:
            top1_avg = meters[0].get_avg()
            top5_avg = meters[1].get_avg()
        loss_avg = meters[2].get_avg()

        for avg in [meters[0], meters[1], meters[2]]:
            avg.reset()

        if system:
            print('[Test] Loss: {loss:.4f}, Top1: {top1: .4f}, Top5: {top5: .4f}'.format(
                loss=loss_avg,
                top1=top1_avg,
                top5=top5_avg
            ))
            return loss_avg, top1_avg, top5_avg
        else:
            print('[Test] Loss: {loss:.4f}'.format(
                loss=loss_avg
            ))
            return loss_avg


if __name__ == '__main__':
    dset = dataset.get_dataset(
        './face_age',
        [0.6361, 0.4875, 0.4189],
        [0.2105, 0.1893, 0.1820],
        target_transform=dataset.face_target
    )
    _, test_set, _, = dataset.train_test_set_split(dset, 'face_age', test_size=0.2)
    test_model = model.load_model('face_age_deploy1')
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, num_workers=2
    )
    meters = [
        utils.AverageMeter(),
        utils.AverageMeter(),
        utils.AverageMeter()
    ]
    criterion = torch.nn.L1Loss()
    loss = test(test_model, test_loader, meters, criterion)
