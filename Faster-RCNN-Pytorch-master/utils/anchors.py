import numpy as np

def generate_anchor_base(side_length=16, ratios=[0.5, 1, 2],
                         scales=[0.5, 1, 2], strides=16):
    """
	Generate anchors for a single 16*16 block. Then transform the anchors

    """
    py = side_length / 2.
    px = side_length / 2.

    anchor_base = np.zeros((len(ratios) * len(scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = side_length * strides * scales[j] * np.sqrt(ratios[i])
            w = side_length * strides * scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def get_anchors(anchor_base, feat_stride, height, width):
    anchors_y = np.arange(height) * feat_stride
    anchors_x = np.arange(width) * feat_stride
    anchors_x, anchors_y = np.meshgrid(anchors_x, anchors_y)
    shift = np.stack((anchors_y.ravel(), anchors_x.ravel(),
                      anchors_y.ravel(), anchors_x.ravel()), axis=1)
    anchors = np.repeat(shift, repeats=len(anchor_base), axis=0) + \
        np.tile(anchor_base, [len(shift),1])
    return anchors

def get_rois_from_loc_anchors(anchors, rpn_locs):
    """
    Decode bounding boxes from bounding box offsets and scales.

    """
    src_bbox = anchors
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = rpn_locs[:, 0]
    dx = rpn_locs[:, 1]
    dh = rpn_locs[:, 2]
    dw = rpn_locs[:, 3]

    dst_y = dy * src_height + src_ctr_y
    dst_x = dx * src_width + src_ctr_x
    dst_h = np.exp(dh) * src_height
    dst_w = np.exp(dw) * src_width

    dst_bbox = np.zeros(rpn_locs.shape, dtype=rpn_locs.dtype)
    dst_bbox[:, 0] = dst_y - 0.5 * dst_h
    dst_bbox[:, 1] = dst_x - 0.5 * dst_w
    dst_bbox[:, 2] = dst_y + 0.5 * dst_h
    dst_bbox[:, 3] = dst_x + 0.5 * dst_w

    return dst_bbox