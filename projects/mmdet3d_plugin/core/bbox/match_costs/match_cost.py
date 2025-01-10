import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        # return bbox_cost * self.weight

        # device = bbox_pred.device
        # dtype = bbox_pred.dtype
        # print("bbox_pred.device:", bbox_pred.device)
        # print("bbox_pred.dtype:", bbox_pred.dtype)
        # bbox_pred = bbox_pred.to(dtype=torch.float32)
        # gt_bboxes = gt_bboxes.to(dtype=torch.float32)
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        # return bbox_cost * self.weight

        # device = bbox_pred.device
        # dtype = bbox_pred.dtype
        # print("bbox_pred.device:", bbox_pred.device)
        # bbox_pred_cpu = bbox_pred.to("cpu", dtype=torch.float32)
        # gt_bboxes_cpu = gt_bboxes.to("cpu", dtype=dtype)
        # bbox_cost_cpu = torch.cdist(bbox_pred_cpu, gt_bboxes_cpu, p=1)
        # bbox_cost = bbox_cost_cpu.to(device, dtype=dtype)
        # return bbox_cost * self.weight

        # device = bbox_pred.device
        # dtype = bbox_pred.dtype
        # print("bbox_pred.device:", bbox_pred.device)
        # bbox_pred = bbox_pred.to("cpu", dtype=dtype)
        # gt_bboxes = gt_bboxes.to("cpu", dtype=dtype)
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        # bbox_cost = bbox_cost.to(device, dtype=dtype)
        # return bbox_cost * self.weight
        def euclidean_distance(X, Y):
            # Broadcasting to calculate squared differences
            diff = X[:, None, :] - Y[None, :, :]
            dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
            return dist
        bbox_cost = euclidean_distance(bbox_pred, gt_bboxes)
        return bbox_cost * self.weight
