import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from torchvision.utils import make_grid
from lib.CATN_R import CATNet as Network
from lib.CATN_R import affinity, edge
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from torch.autograd import Variable

import logging
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name,_ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, _, _ = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

def train(train_loader, model, optimizer, epoch, save_path,test_path):
    """
    train function
    """
    global step
    global best
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            pre40, pre30, pre20, pre10, mean1, mean2, mean3, std1, std2, std3 = model(images)
            loss_seg = structure_loss(pre40, gts) + structure_loss(pre30, gts) + structure_loss(pre20, gts) + structure_loss(pre10, gts)

            # cb_texloss = affinity(mean1, gts, [48]) + affinity(mean2, gts,[24]) + affinity(mean3, gts, [24])
            # std_texloss = affinity(std1, gts, [48]) + affinity(std2, gts, [24]) + affinity(std3, gts, [24])    #384时候使用

            cb_texloss = affinity(mean1, gts, [44]) + affinity(mean2, gts, [22]) + affinity(mean3, gts, [22])
            std_texloss = affinity(std1, gts, [44]) + affinity(std2, gts, [22]) + affinity(std3, gts, [22])

            # micro_loss = (edge(mean1, gts, [12, 24]) + edge(std1,gts, [12, 24]) + edge(mean2, gts, [12, 24]) + edge(std2, gts, [12, 24]))
            micro_loss = (
                        edge(mean1, gts, [11, 22]) + edge(std1, gts, [11, 22]) + edge(mean2, gts, [11, 22]) + edge(std2,
                                                                                                                   gts,
                                                                                                                   [11,
                                                                                                                    22]))

            (loss_seg + cb_texloss + std_texloss + micro_loss).backward()

            # (loss_seg + cb_texloss + std_texloss + micro_loss).backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], seg_loss: {:.4f} macro_loss: {:.4f} micro_loss: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss_seg.data, (cb_texloss + std_texloss).data, micro_loss.data))

        loss_all /= epoch_step
        if (epoch >= 75)and(epoch % 5 == 0):
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))


        global dict_plot

        test1path = './data/TestDataset'
        # if (epoch + 1) % 1 == 0 and epoch >= 75:
        dice = 0
        if (epoch + 1) % 1 == 0 and epoch >= 80:
            for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            # for dataset in ['ETIS-LaribPolypDB']:
                dataset_dice = test(model, test1path, dataset)
                dice += dataset_dice
                logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
                print(dataset, ': ', dataset_dice)
                dict_plot[dataset].append(dataset_dice)
            meandice = dice /5
            # meandice = dataset_dice
            dict_plot['test'].append(meandice)
            if meandice > best:
                best = meandice
                # torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
                torch.save(model.state_dict(), save_path +  'Net_epoch_{}_best.pth'.format(epoch))
                print('##############################################################################best', best)
                logging.info(
                    '##############################################################################best:{}'.format(
                        best))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    plt.show()


# def val(test_loader, model, epoch, save_path):
#     """
#     validation function
#     """
#     global best_mae, best_epoch
#     model.eval()
#     with torch.no_grad():
#         mae_sum = 0
#         for i in range(test_loader.size):
#             image, gt, name, img_for_post = test_loader.load_data()
#             gt = np.asarray(gt, np.float32)
#             gt /= (gt.max() + 1e-8)
#             image = image.cuda()
#
#             res, _, _ = model(image)
#             res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#             mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
#         mae = mae_sum / test_loader.size
#         print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
#         if epoch == 1:
#             best_mae = mae
#         else:
#             if mae < best_mae:
#                 best_mae = mae
#                 best_epoch = epoch
#                 torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
#                 print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

if __name__ == '__main__':
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=101, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./data/TrainDataset',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./data/TestDataset',
                        help='the test rgb images root')

    parser.add_argument('--save_path', type=str,
                        default='./snapshot/CATN_R/',
                        help='the path to save model and log')
    # parser.add_argument('--pth_path', type=str, default='./snapshot/PVT_PCA128/Net_epoch_7_best.pth')
    opt = parser.parse_args()


    # set the device for training
    # if opt.gpu_id == '0':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     print('USE GPU 0')
    # elif opt.gpu_id == '1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #     print('USE GPU 1')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    cudnn.benchmark = True


    model = Network(nf=128)
    # model_dict = model.state_dict()
    # ckpt = torch.load(opt.pth_path)
    # pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict, strict=False)
    model.cuda()
    best = 0

    # model = Network(nf=128)
    # # model.load_state_dict(torch.load(opt.pth_path))
    # model.cuda()
    # best = 0

    # if opt.load is not None:
    #     model.load_state_dict(torch.load(opt.load))
    #     print('load model from ', opt.load)


    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        print("adam")
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, weight_decay=1e-4, momentum=0.9)


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + '/' + 'images/',
                              gt_root=opt.train_root +'/'+ 'masks/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    # val_loader = test_dataset(image_root=opt.val_root + 'images/',
    #                           gt_root=opt.val_root + 'masks/',
    #                           testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0

    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, save_path, opt.val_root)
        # val(val_loader, model, epoch, save_path)
    # plot_train(dict_plot, name)
