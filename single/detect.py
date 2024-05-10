import torch
import numpy as np
import cv2
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, \
    xyxy2xywh
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors
from distance import cal_distance


class Yolov5:
    device = 'cpu'
    weights = 'yolov5/yolov5n.pt'  # model.pt path(s)
    imgsz = 640  # inference size (pixels)

    def __init__(self):
        set_logging()
        self.device = select_device(self.device)

        # Load model
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names

        self.imgsz = [self.imgsz]
        self.imgsz *= 2
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, *self.imgsz, ).to(self.device).type_as(next(self.model.parameters())))  # run once

    @torch.no_grad()
    def run(self,
            im0s,  # HWC图片
            imgsz=640,  # inference size (pixels)
            conf_thres=0.7,  # confidence threshold
            iou_thres=0.7,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            ):

        # Load image
        assert im0s is not None, 'Image Not Available '
        img = letterbox(im0s, imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        dt = [0.0, 0.0, 0.0]
        t1 = time_sync()
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = self.model(img, augment=augment, visualize=visualize)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # Process predictions
        result = []
        det = pred[0]  # per image
        s = ''
        im0 = im0s.copy()
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0  # for save_crop
        # 创建一个用于标注边界框的对象，传入图像、线宽、类别名称等信息
        annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
        if len(det):
            # print('det', det)
            # Rescale boxes from img_size to im0 size
            # 将预测信息映射到原图
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            # 打印检测到的类别数量
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                result.append(xywh)

                c = int(cls)  # integer class
                label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
        im_result = annotator.result()

        # Print time (inference-only)
        print(f'{s}Done. ({t3 - t2:.3f}s)')
        return im_result, result
