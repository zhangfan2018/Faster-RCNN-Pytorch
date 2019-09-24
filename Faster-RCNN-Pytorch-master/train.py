
import numpy as np
from tqdm import tqdm
import torch as t
from torch.autograd import Variable
from torch.utils import data as data_

from utils import array_tool as at
from utils.config import opt
from utils.eval_tool import eval_detection_voc
from utils.vis_tool import vis_bbox

from data.util import read_image
from data.dataset import Dataset, TestDataset

from model.faster_rcnn import FasterRCNN
from trainer.trainer import FasterRCNNTrainer

import warnings
warnings.filterwarnings("ignore")

def train(**kwargs):
    opt._parse(kwargs)

    # 训练数据加载器
    data_set = Dataset(opt)
    data_loader = data_.DataLoader(data_set,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=opt.num_workers)

    # 测试数据加载器
    test_set = TestDataset(opt)
    test_dataloader = data_.DataLoader(test_set,
                                       batch_size=1,
                                       num_workers=1,
                                       shuffle=False,
                                       pin_memory=True)

    # 网络模型
    faster_rcnn = FasterRCNN()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    # 训练过程
    for epoch in range(opt.epoch):

        # eval_result = eval(test_dataloader, faster_rcnn)
        # best_map = eval_result['map']
        loss_list_roi_cls = []
        loss_list_roi_loc = []
        loss_list_rpn_cls = []
        loss_list_rpn_loc = []
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(data_loader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            loss_list = trainer.train_step(img, bbox, label, scale)

            loss_list_roi_cls.append(loss_list.roi_cls_loss.detach().cpu().numpy())
            loss_list_roi_loc.append(loss_list.roi_loc_loss.detach().cpu().numpy())
            loss_list_rpn_cls.append(loss_list.rpn_cls_loss.detach().cpu().numpy())
            loss_list_rpn_loc.append(loss_list.rpn_loc_loss.detach().cpu().numpy())

        print("--------------------------")
        print("curr epoch: ", epoch)
        print("roi_cls loss: ", np.array(loss_list_roi_cls).mean())
        print("roi_loc loss: ", np.array(loss_list_roi_loc).mean())
        print("rpn_cls loss: ", np.array(loss_list_rpn_cls).mean())
        print("rpn_loc loss: ", np.array(loss_list_rpn_loc).mean())
        print("--------------------------")

        eval_result = eval(test_dataloader, faster_rcnn)
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)

        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)

def eval(dataloader, faster_rcnn):

    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == opt.test_num:
            break

    result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores,
                                gt_bboxes, gt_labels, gt_difficults,
                                use_07_metric=True)

    return result

def test():
    img_arr = read_image('demo.jpg')
    img = t.from_numpy(img_arr)[None]

    faster_rcnn = FasterRCNN()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.load('weights/chainer_best_model_converted_to_pytorch_0.7053.pth')
    opt.caffe_pretrain = True
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    vis_bbox(at.tonumpy(img[0]),
             at.tonumpy(_bboxes[0]),
             at.tonumpy(_labels[0]).reshape(-1),
             at.tonumpy(_scores[0]).reshape(-1))

if __name__ == '__main__':
    # train()
    # val()
    test()