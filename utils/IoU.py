import numpy as np

class IoU:
    def __init__(self, premask, groundtruth):
        self.seg_inv = np.logical_not(premask)
        self.gt_inv = np.logical_not(groundtruth)
        self.true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
        self.true_neg = np.logical_and(self.seg_inv, self.gt_inv).sum()
        self.false_pos = np.logical_and(premask, self.gt_inv).sum()
        self.false_neg = np.logical_and(self.seg_inv, groundtruth).sum()

    def get_IoU(self):
        # iou = tp / (tp + fn + fp)
        IoU = self.true_pos / (self.true_pos + self.false_neg + self.false_pos + 1e-6)
        return IoU