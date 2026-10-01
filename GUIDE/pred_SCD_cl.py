import os
import time
import argparse
import numpy as np
import torch.autograd
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
#################################
from datasets import RS_ST as RS

from models.GUIDE import GUIDE as Net

from utils.utils import accuracy, SCDD_eval_all, AverageMeter
import matplotlib.pyplot as plt
import numpy as np
DATA_NAME = 'XXXX'
#################################

class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        
    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='XXXX', help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default='XXXX', help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default='XXXX.pth')

        self.initialized = True
        return parser
        
    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt
        
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')        

def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(3,RS.num_classes-1).cuda()
    net.load_state_dict( torch.load(opt.chkpt_path) )
    net.eval()
    
    test_set = RS.Data_test(opt.test_dir)
    #test_set = RS.Data("val")
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)    
    #predict_direct(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True)
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)

#For models with 3 outputs: 1 change map + 2 semantic maps.
#Parameters: flip->test time augmentation     index_map->"False" means rgb results      intermediate->whether to outputs the intermediate maps
def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    pred_A_dir_rgb_nomask = os.path.join(pred_dir, 'im1_rgb_nomask')
    pred_B_dir_rgb_nomask = os.path.join(pred_dir, 'im2_rgb_nomask')
    pred_changemask_dir_rgb = os.path.join(pred_dir, 'changemask_rgb')

    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if not os.path.exists(pred_A_dir_rgb_nomask): os.makedirs(pred_A_dir_rgb_nomask)
    if not os.path.exists(pred_B_dir_rgb_nomask): os.makedirs(pred_B_dir_rgb_nomask)
    if not os.path.exists(pred_changemask_dir_rgb): os.makedirs(pred_changemask_dir_rgb)

    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    if intermediate:
        pred_mA_dir = os.path.join(pred_dir, 'im1_semantic')
        pred_mB_dir = os.path.join(pred_dir, 'im2_semantic')
        pred_change_dir = os.path.join(pred_dir, 'change')
        if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
        if not os.path.exists(pred_mB_dir): os.makedirs(pred_mB_dir)
        if not os.path.exists(pred_change_dir): os.makedirs(pred_change_dir)
    
    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        #imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad(): 
            out_change,outputs_A, outputs_B,_,_ = net(imgs_A, imgs_B)#,aux
            outputs_A = F.interpolate(outputs_A, size=imgs_B.size()[2:], mode='bilinear', align_corners=True)
            outputs_B = F.interpolate(outputs_B, size=imgs_B.size()[2:], mode='bilinear', align_corners=True)

            out_change = F.sigmoid(out_change)

        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)
            
            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            out_change_v, outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            out_change_v = torch.flip(out_change_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            out_change += F.sigmoid(out_change_v)
            
            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            outputs_A_h, outputs_B_h,out_change_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            out_change_h = torch.flip(out_change_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            out_change += F.sigmoid(out_change_h)
            
            imgs_A_hv = torch.flip(imgs_A, [2,3])
            imgs_B_hv = torch.flip(imgs_B, [2,3])
            out_change_hv, outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
            out_change_hv = torch.flip(out_change_hv, [2,3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            out_change += F.sigmoid(out_change_hv)
            out_change = out_change/4               
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach()>0.5
        change_mask = change_mask.squeeze()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze()+1
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()+1
        
        if intermediate:
            pred_A_path = os.path.join(pred_mA_dir, mask_name)
            pred_B_path = os.path.join(pred_mB_dir, mask_name)
            pred_change_path = os.path.join(pred_change_dir, mask_name)
            io.imsave(pred_A_path, RS.Index2Color(pred_A.numpy()))
            io.imsave(pred_B_path, RS.Index2Color(pred_B.numpy()))
            change_map = exposure.rescale_intensity(change_mask.numpy(), 'image', 'dtype')
            io.imsave(pred_change_path, change_map)
        pred_A_nomask = pred_A
        pred_B_nomask = pred_B
        pred_A = (pred_A*change_mask.long()).numpy()
        pred_B = (pred_B*change_mask.long()).numpy()      
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
        pred_A_path_nomask = os.path.join(pred_A_dir_rgb_nomask, mask_name)
        pred_B_path_nomask = os.path.join(pred_B_dir_rgb_nomask, mask_name)
        pred_changemask_path = os.path.join(pred_changemask_dir_rgb, mask_name)   
        #print(type(pred_A))
        io.imsave(pred_A_path, RS.Index2Color(pred_A))
        io.imsave(pred_B_path, RS.Index2Color(pred_B))
        io.imsave(pred_A_path_nomask, RS.Index2Color(pred_A_nomask))
        io.imsave(pred_B_path_nomask, RS.Index2Color(pred_B_nomask))
        
        io.imsave(pred_changemask_path, change_mask*255)   
        print(pred_A_path)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name)
            pred_B_path = os.path.join(pred_B_dir, mask_name)
            io.imsave(pred_A_path, pred_A.astype(np.uint8))
            io.imsave(pred_B_path, pred_B.astype(np.uint8))  

    
#For models that directly produce 2 SCD maps.
#Parameters: flip->test time augmentation     index_map->"False" means rgb results
def predict_direct(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False,):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if index_map:
        pred_A_dir = os.path.join(pred_dir, 'im1')
        pred_B_dir = os.path.join(pred_dir, 'im2')
        if not os.path.exists(pred_A_dir): os.makedirs(pred_A_dir)
        if not os.path.exists(pred_B_dir): os.makedirs(pred_B_dir)
    preds_all = []
    labels_all = []
    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B,labels_A,labels_B = data
        #imgs = torch.cat([imgs_A, imgs_B], 1)
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad(): 
            outputs_A, outputs_B,out_change = net(imgs_A, imgs_B)#,aux
        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)
            
            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            out_change_v,outputs_A_v, outputs_B_v = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            
            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            out_change_v,outputs_A_h, outputs_B_h = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            
            imgs_A_hv = torch.flip(imgs_A, [2,3])
            imgs_B_hv = torch.flip(imgs_B, [2,3])
            out_change_v,outputs_A_hv, outputs_B_hv = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2,3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2,3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        labels_A = labels_A.cpu().detach().numpy()
        
        labels_B = labels_B.cpu().detach().numpy()
        pred_A = torch.argmax(outputs_A, dim=1)
        pred_B = torch.argmax(outputs_B, dim=1)
        pred_A = pred_A.numpy().astype(np.uint8)
        pred_B = pred_B.numpy().astype(np.uint8)
        for (pred1_A, pred1_B, label_A, label_B) in zip(pred_A, pred_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred1_A, label_A)
            acc_B, valid_sum_B = accuracy(pred1_B, label_B)
            preds_all.append(pred1_A)
            preds_all.append(pred1_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            # acc = (acc_A + acc_B)*0.5
            # acc_meter.update(acc)
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
        # io.imsave(pred_A_path, RS.Index2Color(pred_A))
        # io.imsave(pred_B_path, RS.Index2Color(pred_B))
        # print(pred_A_path)
        if index_map:
            pred_A_path = os.path.join(pred_A_dir, mask_name)
            pred_B_path = os.path.join(pred_B_dir, mask_name)
            # io.imsave(pred_A_path, pred_A.astype(np.uint8))
            # io.imsave(pred_B_path, pred_B.astype(np.uint8))
        '''
        change_path = os.path.join(pred_dir, 'change', mask_name)
        io.imsave(change_path, (change_mask*255).astype(np.uint8))'''
    # Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)
    kappa, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)
    print('kappa: %.1f, IOU: %.2f  sek %.2f ' %(kappa*100, IoU_mean*100, Sek*100))




if __name__ == '__main__':
    main()