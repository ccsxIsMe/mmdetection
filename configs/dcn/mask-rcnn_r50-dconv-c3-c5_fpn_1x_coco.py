_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco1.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
