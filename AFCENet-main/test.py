from util.metric_tool import ConfuseMatrixMeter
import torch
from option import Options
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt  
import sys
import argparse
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from PIL import Image
import time

def _colorize_tp_fp_fn(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    tp = pred & gt
    fp = pred & (~gt)
    fn = (~pred) & gt
    out[tp] = (255, 255, 255)
    out[fp] = (255, 0, 0)
    out[fn] = (0, 255, 0)
    return out


def _to_uint8_rgb_from_imagenet_norm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    x = (x * 255.0).byte()
    return x.permute(1, 2, 0).cpu().numpy()


def _make_compare_mosaic(t1_rgb: np.ndarray, t2_rgb: np.ndarray, gt_mask: np.ndarray, err_rgb: np.ndarray) -> Image.Image:
    h, w = gt_mask.shape
    gt_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    gt_rgb[gt_mask.astype(bool)] = (255, 255, 255)
    gt_pil = Image.fromarray(gt_rgb, mode='RGB')
    t1_pil = Image.fromarray(t1_rgb, mode='RGB')
    t2_pil = Image.fromarray(t2_rgb, mode='RGB')
    err_pil = Image.fromarray(err_rgb, mode='RGB')

    mosaic = Image.new('RGB', (w * 4, h))
    mosaic.paste(t1_pil, (0, 0))
    mosaic.paste(t2_pil, (w, 0))
    mosaic.paste(gt_pil, (w * 2, 0))
    mosaic.paste(err_pil, (w * 3, 0))
    return mosaic


if __name__ == '__main__':

    # 额外的测试参数（不影响 Options() 原有参数）
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_name', type=str, default=None, help='checkpoints 下的子目录名，如 LEVIR_20260101_203500')
    parser.add_argument('--save_pred', action='store_true', help='是否保存预测图')
    test_args, remaining = parser.parse_known_args()

    # 临时修改 sys.argv，让 Options 只解析剩余参数
    sys.argv = [sys.argv[0]] + remaining

    opt = Options().parse()
    opt.phase = 'test'

    # 指定 checkpoint 子目录（Model.save_dir = checkpoint_dir/name）
    if test_args.ckpt_name:
        opt.name = test_args.ckpt_name
        print(f"使用checkpoint: {opt.name}")
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)
    checkpoint_save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    tbar = tqdm(test_data, ncols=80)
    total_iters = test_size
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    # 创建结果保存目录
    if test_args.save_pred:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(project_dir, 'results', opt.name)
        pred_dir = os.path.join(save_dir, 'pred')
        err_dir = os.path.join(save_dir, 'error')
        compare_dir = os.path.join(save_dir, 'compare')
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(err_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)
        print(f"预测图将保存到: {save_dir}")
    
    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
            # update metric
            val_target = _data['cd_label'].detach()
            val_pred_argmax = torch.argmax(val_pred.detach(), dim=1)
            _ = running_metric.update_cm(pr=val_pred_argmax.cpu().numpy(), gt=val_target.cpu().numpy())
            
            # 保存预测结果为可视化图像
            if test_args.save_pred:
                for j in range(val_pred_argmax.shape[0]):
                    fname = _data['fname'][j]
                    pred_mask = val_pred_argmax[j].cpu().numpy().astype(np.uint8)
                    gt_mask = val_target[j].cpu().numpy().astype(np.uint8)

                    pred_img = pred_mask * 255
                    pred_pil = Image.fromarray(pred_img, mode='L')
                    pred_path = os.path.join(pred_dir, fname)
                    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                    pred_pil.save(pred_path)

                    err_rgb = _colorize_tp_fp_fn(pred_mask, gt_mask)
                    err_pil = Image.fromarray(err_rgb, mode='RGB')
                    err_path = os.path.join(err_dir, fname)
                    os.makedirs(os.path.dirname(err_path), exist_ok=True)
                    err_pil.save(err_path)

                    t1_rgb = _to_uint8_rgb_from_imagenet_norm(_data['img1'][j].cpu())
                    t2_rgb = _to_uint8_rgb_from_imagenet_norm(_data['img2'][j].cpu())
                    mosaic = _make_compare_mosaic(t1_rgb, t2_rgb, gt_mask, err_rgb)
                    compare_path = os.path.join(compare_dir, fname)
                    os.makedirs(os.path.dirname(compare_path), exist_ok=True)
                    mosaic.save(compare_path)
                
        val_scores = running_metric.get_scores()
        message = '(phase: %s) ' % (opt.phase)
        for k, v in val_scores.items():
            message += '%s: %.3f ' % (k, v * 100)
        print(message)
        log_path = os.path.join(checkpoint_save_dir, 'log.txt')
        with open(log_path, 'a') as log_file:
            log_file.write('[%s] %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), message))

"""
(phase: test) acc: 99.436 miou: 97.446 mf1: 98.697 iou_0: 99.358 iou_1: 95.533 F1_0: 99.678 F1_1: 97.716 
precision_0: 99.687 precision_1: 97.654 recall_0: 99.669 recall_1: 97.778 

Prec. 0.9765 Recall 0.9778 F1 0.9772 OA 0.9944

SYSU_mobilenetv2_best
"""
