_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco1.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=False))
