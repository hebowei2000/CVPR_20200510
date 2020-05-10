import argparse
import sys
from datetime import datetime

import imageio

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from util import *
from net import *


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


def train(train_loader, model, optimizer, args):
    # multi resolution
    size_rates = [1]

    args.log_path = '../log/' + args.model
    save_path = '../model/' + args.model
    sw = SummaryWriter(args.log_path)

    focal_loss = FocalLoss()

    loss_sal_record, loss_l2_record, loss_dice_record, loss_focal_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    total_step = len(train_loader)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    global_step = 0
    for epoch in range(0, args.epoch):
        model.train()

        for step, data in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                ims, gts, names = data
                # Load data
                ims = ims.cuda()
                gts = gts.cuda()
                # Forward
                trainsize = int(round(args.train_size * rate / 32) * 32)
                if rate != 1:
                    ims = F.upsample(ims, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                pred_sal = model(ims)

                loss_sal = nn.BCEWithLogitsLoss()(pred_sal, gts)
                loss_l2 = nn.MSELoss()(pred_sal.sigmoid(), gts)
                loss_dice = DiceLoss()(pred_sal.sigmoid(), gts)
                # loss_ohem = OhemCrossEntropy(pred_sal, gts)
                # loss_focal = FocalLoss()(pred_sal, gts)
                loss = loss_sal + loss_l2 + loss_dice * 0.1
                loss.backward()

                optimizer.step()
                if rate == 1:
                    loss_sal_record.update(loss_sal.data, args.batch_size)
                    loss_l2_record.update(loss_l2.data, args.batch_size)
                    loss_dice_record.update(loss_dice.data, args.batch_size)
                    # loss_focal_record.update(loss_focal.data, args.batch_size)

            sw.add_scalar('lr', scheduler.get_lr()[0], global_step=global_step)
            sw.add_scalars('loss', {'BCELoss': loss_sal_record.show(), 'MSELoss': loss_l2_record.show(),
                                    'DiceLoss': loss_dice_record.show(), 'FocalLoss': 0.},
                           global_step=global_step)

            if step % 10 == 0 or step == total_step:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, BCELoss: {:.4f}, MSELoss: {:.4f}, DiceLoss: {:.4f}, FocalLoss: {:.4f}'.
                        format(datetime.now(), epoch, args.epoch, step, total_step, scheduler.get_lr()[0],
                               loss_sal_record.show(), loss_l2_record.show(), loss_dice_record.show(),
                               0.), flush=True)
            global_step += 1

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch >= 20:
            torch.save(model.state_dict(), save_path + args.model + '_' + '.%d' % epoch + '.pth')

        scheduler.step()

    test(model, -1, args)


def test(model, epoch, args):
    model.eval()
    for dataset in args.valset:
        save_path = './out/' + args.model + '/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = args.data_path + dataset + '/'
        gt_root = args.data_path + '/gt/'
        test_loader = PixelDataset(image_root, gt_root, args.train_size)
        for i in range(test_loader.size):
            image, name = test_loader.load_data()
            image = image.cuda()
            attention = model(image)
            attention = F.upsample(attention, size=(256, 256), mode='bilinear', align_corners=True)
            res = attention.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imsave(save_path + name + '.png', res)


def main():
    # init parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--train_size', type=int, default=256, help='training dataset size')
    parser.add_argument('--trainset', type=str, default='DUTS_TRAIN', help='training  dataset')
    parser.add_argument('--channel', type=int, default=30, help='channel number of convolutional layers in decoder')
    parser.add_argument('--is_resnet', type=bool, default=True, help='VGG or ResNet backbone')
    parser.add_argument('--model', type=str, default='baseline', help='VGG or ResNet backbone')
    args = parser.parse_args()

    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)

    print('Learning Rate: {} ResNet: {} Trainset: {}'.format(args.lr, args.is_resnet, args.trainset))

    # build model
    model = globals()[args.model]()
    model = model.cuda()

    params = model.parameters()
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(params, args.lr, weight_decay=5e-4)

    # dataset
    args.data_path = '../data/pixel/'
    image_root = args.data_path + 'train_new/'
    gt_root = args.data_path + 'gt_new/'
    train_loader = pixel_loader(image_root, gt_root, args.batch_size, args.train_size)
    args.valset = ['test']

    # begin training
    print("Time to witness the mirracle!")
    train(train_loader, model, optimizer, args)


if __name__ == '__main__':
    main()
