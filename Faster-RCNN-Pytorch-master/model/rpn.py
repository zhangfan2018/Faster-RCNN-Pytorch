import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

from utils.anchors import generate_anchor_base, get_anchors, get_rois_from_loc_anchors
from utils.py_nms import py_cpu_nms


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
                 self, in_channels=512,
                 mid_channels=512,
                 ratios=[0.5, 1, 2],
                 scales=[0.5, 1, 2],
                 feat_stride=16
    ):
        super(RegionProposalNetwork, self).__init__()
        # prepare anchor base
        self.anchor_base = generate_anchor_base(side_length=16, 
                                                ratios=ratios,
                                                scales=scales,
                                                strides=feat_stride)
        self.feat_stride = feat_stride
        # network params
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, h, img_size, scale=1.):
        """

        Forward Region Proposal Network.

        """
        n_pre_nms = 12000
        n_post_nms = 2000
        nms_thresh = 0.7

        # get anchors predifined
        n, _, hh, ww = h.shape
        anchors = get_anchors(self.anchor_base, self.feat_stride, hh, ww)

        # main forward
        hidd = F.relu(self.conv1(h))
        rpn_locs = self.loc(hidd)
        rpn_scores = self.score(hidd)

        # rpn_locs, rpn_scores
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        scores = rpn_scores[:, :, 1].data.cpu().numpy()[0]
        
        # get rois, roi_indices
        rois = get_rois_from_loc_anchors(anchors, rpn_locs[0].data.cpu().numpy())

        # clip
        rois[:, ::2] = np.clip(rois[:, ::2], 0, img_size[0])
        rois[:, 1::2] = np.clip(rois[:, 1::2], 0, img_size[1])

        # remove < min_size
        min_size = 16
        min_size = min_size * scale
        hs = rois[:, 2] - rois[:, 0]
        ws = rois[:, 3] - rois[:, 1]

        if t.is_tensor(min_size):
            min_size = min_size.numpy()

        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        rois = rois[keep, :]
        scores = scores[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = scores.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        rois = rois[order, :]

        # NMS
        keep = py_cpu_nms(rois, nms_thresh)
        keep = keep[:n_post_nms]
        rois = rois[keep]
        return rpn_locs, rpn_scores, rois, [0]*len(rois), anchors


def normal_init(m, mean, stddev, truncated=False):
    """

    weight initalizer: truncated normal and random normal.

    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()