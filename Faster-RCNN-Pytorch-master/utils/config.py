from pprint import pprint

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/mnt/sda1/temp_workspace/Faster-RCNN-Pytorch/Faster-RCNN-Pytorch-master/datasets/VOC2007'
    min_size = 600
    max_size = 1000
    num_workers = 1
    test_num = 10000

    # sigma weight for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    lr = 1e-3
    weight_decay = 0.0005

    # training param
    epoch = 14
    use_adam = False

    # caffe model config
    caffe_pretrain = False
    caffe_pretrain_path = 'weights/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth'
    load_path = 'weights/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()
