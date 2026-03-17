tensor([[ 54,  54,   0,  ...,  99,  99,  99],
        [ 54,  54,  54,  ...,  99,  99,  99],
        [ 54,  54,  54,  ...,  99,  99,  99],
        ...,
        [203, 203, 203,  ...,   1,   1,   1],
        [203, 203, 203,  ...,   1,   1,   1],
        [203, 203, 203,  ...,   1,   1,   1]], device='cuda:0')


tensor([[0.5511, 0.5729, 0.6404,  ..., 0.9986, 0.9967, 0.9944],
        [0.8614, 0.9377, 0.7940,  ..., 0.9960, 0.9921, 0.9906],
        [0.9393, 0.9902, 0.9143,  ..., 0.9919, 0.9898, 0.9898],
        ...,
        [0.9610, 0.9640, 0.9710,  ..., 1.0000, 1.0000, 1.0000],
        [0.9540, 0.9583, 0.9665,  ..., 1.0000, 1.0000, 1.0000],
        [0.9433, 0.9534, 0.9598,  ..., 1.0000, 1.0000, 1.0000]],
       device='cuda:0')

# boundary weighted  + self confidence
classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')

gt_obj = viewpoint_cam.objects.cuda().long()

logits = classifier(objects)

loss_map = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze(0)

# ---------------- boundary weight ----------------

import torch.nn.functional as F

gt = gt_obj.unsqueeze(0).unsqueeze(0).float()

dilated = F.max_pool2d(gt, 5, stride=1, padding=2)
eroded = -F.max_pool2d(-gt, 5, stride=1, padding=2)

boundary = (dilated != eroded).squeeze()

weight_map = torch.ones_like(loss_map)
weight_map[boundary] = 0.3

# ---------------- self confidence ----------------

with torch.no_grad():
    prob = torch.softmax(logits, dim=1)

    gt_prob = torch.gather(
        prob,
        1,
        gt_obj.unsqueeze(0).unsqueeze(0)
    ).squeeze()

    self_weight = torch.clamp(gt_prob, min=0.05, max=1.0)

weight_map = weight_map * self_weight

# ---------------- weighted loss ----------------

loss_obj = (loss_map * weight_map).sum() / (weight_map.sum() + 1e-6)

loss_obj = loss_obj / torch.log(torch.tensor(num_classes).cuda())