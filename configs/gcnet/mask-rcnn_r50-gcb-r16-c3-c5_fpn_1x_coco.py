_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco1.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            stages=(False, True, True, True),
            position='after_conv3')
    ]))
