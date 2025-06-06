_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco1.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

# MMEngine support the following two ways, users can choose
# according to convenience
# optim_wrapper = dict(type='AmpOptimWrapper')
_base_.optim_wrapper.type = 'AmpOptimWrapper'
