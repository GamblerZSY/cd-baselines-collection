import os
import cv2
import numpy as np
from tqdm import tqdm
from functools import partial
from sklearn import metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Meter:
    def __init__(self, callback=None, calc_avg=True):
        super().__init__()
        if callback is not None:
            self.calculate = callback
        self.calc_avg = calc_avg
        self.reset()

    def calculate(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            raise ValueError

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        if self.calc_avg:
            self.avg = 0

    def update(self, *args, n=1):
        self.val = self.calculate(*args)
        self.sum += self.val * n
        self.count += n
        if self.calc_avg:
            self.avg = self.sum / self.count

    def __repr__(self):
        if self.calc_avg:
            return "val: {} avg: {} cnt: {}".format(self.val, self.avg, self.count)
        else:
            return "val: {} cnt: {}".format(self.val, self.count)


# These metrics only for numpy arrays
class Metric(Meter):
    __name__ = 'Metric'
    def __init__(self, n_classes=2, mode='separ', reduction='binary'):
        self._cm = Meter(partial(metrics.confusion_matrix, labels=np.arange(n_classes)), False)
        self.mode = mode
        if reduction == 'binary' and n_classes != 2:
            raise ValueError("Binary reduction only works in 2-class cases.")
        self.reduction = reduction
        super().__init__(None, mode!='accum')
    
    def _calculate_metric(self, cm):
        raise NotImplementedError

    def calculate(self, pred, true, n=1):
        self._cm.update(true.ravel(), pred.ravel())
        if self.mode == 'accum':
            cm = self._cm.sum
        elif self.mode == 'separ':
            cm = self._cm.val
        else:
            raise ValueError("Invalid working mode")
        
        if self.reduction == 'none':
            # Do not reduce size
            return self._calculate_metric(cm)
        elif self.reduction == 'mean':
            # Macro averaging
            return self._calculate_metric(cm).mean()
        elif self.reduction == 'binary':
            # The pos_class be 1
            return self._calculate_metric(cm)[1]
        else:
            raise ValueError("Invalid reduction type")

    def reset(self):
        super().reset()
        # Reset the confusion matrix
        self._cm.reset()

    def __repr__(self):
        return self.__name__+" "+super().__repr__()


class Precision(Metric):
    __name__ = 'Prec.'
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=0))


class Recall(Metric):
    __name__ = 'Recall'
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=1))

 
class Accuracy(Metric):
    __name__ = 'OA'
    def __init__(self, n_classes=2, mode='separ'):
        super().__init__(n_classes=n_classes, mode=mode, reduction='none')
        
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm).sum()/cm.sum())


class F1Score(Metric):
    __name__ = 'F1'
    def _calculate_metric(self, cm):
        prec = np.nan_to_num(np.diag(cm)/cm.sum(axis=0))
        recall = np.nan_to_num(np.diag(cm)/cm.sum(axis=1))
        return np.nan_to_num(2*(prec*recall) / (prec+recall))


class MIoU(Metric):
    __name__ = 'mIoU'
    def _calculate_metric(self, cm):
        # 计算每个类别的IoU：对角线元素 / (行和 + 列和 - 对角线元素)
        iou = np.nan_to_num(np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)))
        return iou


# Levir:Prec. 0.9276 Recall 0.9070 F1 0.9172 OA 0.9917
# gt_path = '/home/cver/4TDisk/hjl/rs_data/LEVIR-CD256/test/label'
# pre_path = 'results/LEVIR' 

#  Prec. 0.9765 Recall 0.9778 F1 0.9772 OA 0.9944
gt_path = '/home/inspur/rs_data/SYSU-CD256/test/label'
pre_path = 'results/SYSU' 


# 初始化进度条
pb = tqdm(os.listdir(pre_path))

desc = ""
# 添加MIoU到指标列表
metrics = (Precision(mode='accum'), Recall(mode='accum'), F1Score(mode='accum'), 
           Accuracy(mode='accum'), MIoU(mode='accum'))

for i, filename in enumerate(pb, start=1):
    pre = os.path.join(pre_path, filename)
    gt = os.path.join(gt_path, filename)

    pred = cv2.imread(pre, cv2.IMREAD_GRAYSCALE).astype('uint8')
    gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE).astype('uint8')

    # 转换为二值图像 (0 和 1)
    _, pred = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)
    _, gt = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

    # 转换为 (1, 256, 256)
    pred = np.expand_dims(pred, axis=0)
    gt = np.expand_dims(gt, axis=0)


    for m in metrics:
        m.update(pred, gt, n=1)

    # 更新描述
    desc = ""
    for m in metrics:
        desc += " {} {:.4f}".format(m.__name__, m.val)
    
    # 设置进度条的描述
    pb.set_description(desc)