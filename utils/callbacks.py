import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import BBoxUtility
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, anchors, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net                = net
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period

        self.anchors            = torch.from_numpy(self.anchors).type(torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()

        self.bbox_util = BBoxUtility(self.num_classes)

        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------#
        #   锟斤拷锟斤拷锟斤拷锟斤拷图片锟侥高和匡拷
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   锟斤拷锟斤拷锟斤将图锟斤拷转锟斤拷锟斤拷RGB图锟今，凤拷止锟揭讹拷图锟斤拷预锟斤拷时锟斤拷锟斤拷锟斤拷
        #   锟斤拷锟斤拷锟斤拷锟街э拷锟絉GB图锟斤拷锟皆わ拷猓拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷偷锟酵硷拷穸蓟锟阶拷锟斤拷锟絉GB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   锟斤拷图锟斤拷锟斤拷锟接伙拷锟斤拷锟斤拷实锟街诧拷失锟斤拷锟絩esize
        #   也锟斤拷锟斤拷直锟斤拷resize锟斤拷锟斤拷识锟斤拷
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   锟斤拷锟斤拷锟斤拷batch_size维锟饺ｏ拷图片预锟斤拷锟斤拷锟斤拷锟斤拷一锟斤拷锟斤拷
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #---------------------------------------------------#
            #   转锟斤拷锟斤拷torch锟斤拷锟斤拷式
            #---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   锟斤拷图锟斤拷锟斤拷锟斤拷锟斤拷锟界当锟叫斤拷锟斤拷预锟解！
            #---------------------------------------------------------#
            outputs     = self.net(images)
            #-----------------------------------------------------------#
            #   锟斤拷预锟斤拷锟斤拷锟斤拷锟叫斤拷锟斤拷
            #-----------------------------------------------------------#
            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   锟斤拷锟矫伙拷屑锟解到锟斤拷锟藉，锟津返伙拷原图
            #--------------------------------------#
            if len(results[0]) <= 0:
                return

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_label)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   锟斤拷取图锟斤拷转锟斤拷锟斤拷RGB图锟斤拷
                #------------------------------#
                image       = Image.open(line[0])
                #------------------------------#
                #   锟斤拷锟皆わ拷锟斤拷
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                #------------------------------#
                #   锟斤拷锟皆わ拷锟絫xt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

                #------------------------------#
                #   锟斤拷锟斤拷锟绞碉拷锟絫xt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
