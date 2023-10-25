import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage

from source.tools.Tool import Tool
from source import utils

import os
import numpy as np

try:
    import torch
    from torch.nn.functional import interpolate
except Exception as e:
    print("Incompatible version between pytorch, cuda and python.\n" +
          "Knowing working version combinations are\n: Cuda 10.0, pytorch 1.0.0, python 3.6.8" + str(e))

from models.dataloaders import helpers as helpers

from segment_anything import SamPredictor
from segment_anything import sam_model_registry


class SAMPredictor(Tool):
    def __init__(self, viewerplus, pick_points):
        super(SAMPredictor, self).__init__(viewerplus)
        self.pick_points = pick_points

        self.CROSS_LINE_WIDTH = 2
        self.pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.red, 'size': 6}
        self.sampredictor_net = None
        self.device = None

    def leftPressed(self, x, y, mods):
        points = self.pick_points.points

        if len(points) < 1 and mods == Qt.ShiftModifier:
            self.pick_points.addPoint(x, y, self.pick_style)
            message = "[TOOL][SAMPREDICTOR] New point picked (" + str(len(points)) + ")"
            self.log.emit(message)

        # APPLY SAM PREDICTOR
        if len(points) == 1:
            self.segmentWithSAMPredictor()
            self.pick_points.reset()

    def prepareForSAMPredictor(self, four_points, pad_max):
        """
        Crop the image map (QImage) and return a NUMPY array containing it.
        It returns also the coordinates of the bounding box on the cropped image.
        """

        left = four_points[:, 0].min() - pad_max
        right = four_points[:, 0].max() + pad_max
        top = four_points[:, 1].min() - pad_max
        bottom = four_points[:, 1].max() + pad_max
        h = bottom - top
        w = right - left

        image_cropped = utils.cropQImage(self.viewerplus.img_map, [top, left, w, h])

        fmt = image_cropped.format()
        assert (fmt == QImage.Format_RGB32)

        arr = np.zeros((h, w, 3), dtype=np.uint8)

        bits = image_cropped.bits()
        bits.setsize(int(h * w * 4))
        arrtemp = np.frombuffer(bits, np.uint8).copy()
        arrtemp = np.reshape(arrtemp, [h, w, 4])
        arr[:, :, 0] = arrtemp[:, :, 2]
        arr[:, :, 1] = arrtemp[:, :, 1]
        arr[:, :, 2] = arrtemp[:, :, 0]

        # update four point
        four_points_updated = np.zeros((4, 2), dtype=np.int32)
        four_points_updated[:, 0] = four_points[:, 0] - left
        four_points_updated[:, 1] = four_points[:, 1] - top

        return (arr, four_points_updated)

    def segmentWithSAMPredictor(self):

        if not self.viewerplus.img_map:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation begins..")

        # load network if necessary
        self.loadNetwork()

        pad = 50

        point = self.pick_points.points[0].tolist()
        extreme_points_to_use = np.array([[point[0] - 200, point[1] - 200],
                                          [point[0] - 200, point[1] + 200],
                                          [point[0] + 200, point[1] - 200],
                                          [point[0] + 200, point[1] + 200]])
        pad_extreme = 100

        left_map_pos = extreme_points_to_use[:, 0].min() - pad_extreme
        top_map_pos = extreme_points_to_use[:, 1].min() - pad_extreme

        width_extreme_points = extreme_points_to_use[:, 0].max() - extreme_points_to_use[:, 0].min()
        height_extreme_points = extreme_points_to_use[:, 1].max() - extreme_points_to_use[:, 1].min()
        area_extreme_points = (width_extreme_points * height_extreme_points) / 4

        (img, extreme_points_new) = self.prepareForSAMPredictor(extreme_points_to_use, pad_extreme)

        with torch.no_grad():

            extreme_points_ori = extreme_points_new.astype(int)

            #  Crop image to the bounding box from the extreme points and resize
            bbox = helpers.get_bbox(img, points=extreme_points_ori, pad=pad, zero_pad=True)
            crop_image = helpers.crop_from_bbox(img, bbox, zero_pad=True)
            resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.uint8)

            #  Generate extreme point heat map normalized to image values
            extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]),
                                                   np.min(extreme_points_ori[:, 1])] + [pad, pad]

            # Set the resized image
            self.sampredictor_net.set_image(resize_image)

            # Grab the point
            input_point = np.expand_dims(np.mean(extreme_points, axis=0), axis=0).astype(np.uint8)
            input_label = np.array([1])

            # Make prediction
            mask, score, logit = self.sampredictor_net.predict(point_coords=input_point,
                                                               point_labels=input_label,
                                                               multimask_output=False)
            mask = mask.squeeze().astype(float)
            segm_mask = helpers.crop2fullmask(mask,
                                              bbox,
                                              im_size=img.shape[:2],
                                              zero_pad=True,
                                              relax=pad).astype(int)

            # TODO: move this function to blob!!!
            blobs = self.viewerplus.annotations.blobsFromMask(segm_mask,
                                                              left_map_pos,
                                                              top_map_pos,
                                                              area_extreme_points)

            self.viewerplus.resetSelection()

            for blob in blobs:
                self.viewerplus.addBlob(blob, selected=True)
                self.blobInfo.emit(blob, "[TOOL][SAMPREDICTOR][BLOB-CREATED]")
            self.viewerplus.saveUndo()

            self.infoMessage.emit("Segmentation done.")

        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation ends.")

        QApplication.restoreOverrideCursor()

    def loadNetwork(self):

        if self.sampredictor_net is None:
            self.infoMessage.emit("Loading SAM network..")

            # Mapping between the model type, and the checkpoint file name
            sam_dict = {"vit_b": "sam_vit_b_01ec64",
                        "vit_l": "sam_vit_l_0b3195",
                        "vit_h": "sam_vit_h_4b8939"}

            # Initialization
            model_type = 'vit_b'
            modelName = sam_dict[model_type]
            models_dir = os.path.join(self.viewerplus.taglab_dir, "models")
            path = os.path.join(models_dir, modelName + '.pth')

            device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

            # Loading the mode, returning the predictor
            sam_model = sam_model_registry[model_type](checkpoint=path)
            sam_model.to(device=device)
            self.sampredictor_net = SamPredictor(sam_model)
            self.device = device

    def resetNetwork(self):

        torch.cuda.empty_cache()
        if self.sampredictor_net is not None:
            del self.sampredictor_net
            self.sampredictor_net = None

    def reset(self):
        self.resetNetwork()
        self.pick_points.reset()