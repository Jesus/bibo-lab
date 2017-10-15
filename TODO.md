[x] Ensure that images without annotations are excluded from the
    training/eval data sets.
[x] Resize images before creating the TF record files.
[x] Try to use a DNN with higher mAP, we've failed to train with
    `faster_rcnn_inception_resnet_v2_atrous_coco`, but got the following error:
     https://stackoverflow.com/q/46135528/814224
