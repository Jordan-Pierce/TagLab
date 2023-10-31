import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPen, QBrush

from source.tools.Tool import Tool
from source import utils

import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from models.dataloaders import helpers as helpers


class SAMPredictor(Tool):
    def __init__(self, viewerplus, pick_points):
        super(SAMPredictor, self).__init__(viewerplus)
        self.pick_points = pick_points

        # Image is resized to
        self.resize_to = 2048
        # Padding amount
        self.pad = 0
        # Model Type (b, l, or h)
        self.sam_model_type = 'vit_l'
        # Mask score threshold
        self.score_threshold = 0.80
        # For debugging
        self.debug = False

        # Mosaic dimensions
        self.width = None
        self.height = None

        # SAM, CUDA or CPU
        self.sampredictor_net = None
        self.device = None

        # Drawing on GUI
        self.CROSS_LINE_WIDTH = 2
        self.pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.red, 'size': 6}
        self.work_area_bbox = None
        self.work_area_item = None

    def leftPressed(self, x, y, mods):

        # Load Network in the beginning
        self.loadNetwork()

        # If the weights are there, continue
        if self.sampredictor_net:

            points = self.pick_points.points

            # Single Click
            # There are no existing points, but this point and shift are clicked
            if not points and mods == Qt.ShiftModifier:
                self.pick_points.addPoint(x, y, self.pick_style)
                message = "[TOOL][SAMPREDICTOR] New point picked (" + str(len(points)) + ")"
                self.log.emit(message)
                # Update working area
                self.getPadding()
                self.getWorkArea()
                # Segment with SAM
                self.segmentWithSAMPredictor()
                self.pick_points.reset()

            # Multi Click
            # Point is clicked without shift
            if mods != Qt.ShiftModifier:
                self.pick_points.addPoint(x, y, self.pick_style)
                message = "[TOOL][SAMPREDICTOR] New point picked (" + str(len(points)) + ")"
                self.log.emit(message)
                # Update working area
                self.getPadding()
                self.getWorkArea()

            # Last Click
            # There are existing points, and the latest point was clicked with shift
            if len(points) and mods == Qt.ShiftModifier:
                self.pick_points.addPoint(x, y, self.pick_style)
                message = "[TOOL][SAMPREDICTOR] New point picked (" + str(len(points)) + ")"
                self.log.emit(message)
                # Update working area
                self.getPadding()
                self.getWorkArea()
                # Segment with SAM
                self.segmentWithSAMPredictor()
                self.pick_points.reset()

    def getPadding(self):
        """
        Get the padding amount based on the location of point(s)
        """

        # Point(s) passed from GUI
        points = np.asarray(self.pick_points.points).astype(int)

        # The amount to pad in all directions around the point(s)
        # Useful as mosaics are of different sizes and fixed values
        # would lead to bad results depending on the mosaic
        if len(points) == 1 and np.max([self.width, self.height]) < 16000:
            # If the mosaic is small, then we need to make the padding bigger
            # when provided a single point as the ideal bbox size is unknown
            pad = int(np.max([self.width, self.height]) * 0.1)
        else:
            # If there are multiple points, then the ideal bbox size is
            # calculated based on the points, plus a small amount of padding
            pad = int(np.max([self.width, self.height]) * 0.05)

        self.pad = pad

    def getWorkArea(self):
        """
        Set the work area based on the location of point(s) and padding
        """
        self.resetWorkArea()

        points = np.asarray(self.pick_points.points).astype(int)

        left = points[:, 0].min() - self.pad
        right = points[:, 0].max() + self.pad
        top = points[:, 1].min() - self.pad
        bottom = points[:, 1].max() + self.pad
        h = bottom - top
        w = right - left

        self.work_area_bbox = [round(top), round(left), round(w), round(h)]

        # Display to GUI
        brush = QBrush(Qt.NoBrush)
        pen = QPen(Qt.DashLine)
        pen.setWidth(2)
        pen.setColor(Qt.white)
        pen.setCosmetic(True)
        x = self.work_area_bbox[1]
        y = self.work_area_bbox[0]
        w = self.work_area_bbox[2]
        h = self.work_area_bbox[3]
        self.work_area_item = self.viewerplus.scene.addRect(x, y, w, h, pen, brush)
        self.work_area_item.setZValue(3)

    def prepareForSAMPredictor(self):
        """
        Get the image based on point(s) location
        """
        points = np.asarray(self.pick_points.points).astype(int)

        left = points[:, 0].min() - self.pad
        right = points[:, 0].max() + self.pad
        top = points[:, 1].min() - self.pad
        bottom = points[:, 1].max() + self.pad
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

        # update points to be in image_cropped coordinate space
        updated_points = np.zeros((len(points), 2), dtype=np.int32)
        updated_points[:, 0] = points[:, 0] - left
        updated_points[:, 1] = points[:, 1] - top

        return arr, updated_points

    def segmentWithSAMPredictor(self):

        if not self.viewerplus.img_map:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation begins..")

        # Mosaic dimensions
        self.width = self.viewerplus.img_map.size().width()
        self.height = self.viewerplus.img_map.size().height()

        # User defined points in GUI
        points = np.asarray(self.pick_points.points).astype(int)

        # Top-left corner of work area in GUI
        left_map_pos = points[:, 0].min() - self.pad
        top_map_pos = points[:, 1].min() - self.pad

        # Image from work area, and points w/ transformed coordinates
        (img, points_ori) = self.prepareForSAMPredictor()

        # Points in img coordinate space
        points_ori = points_ori.astype(int)
        #  Padding of points by amount pad
        bbox = helpers.get_bbox(img, points=points_ori, pad=self.pad, zero_pad=True)
        # Cropping the image, and resizing it
        image_cropped = helpers.crop_from_bbox(img, bbox, zero_pad=True)
        image_resized = helpers.fixed_resize(image_cropped, (self.resize_to, self.resize_to)).astype(np.uint8)

        # Generate points normalized to image values
        points_resized = points_ori - [np.min(points_ori[:, 0]), np.min(points_ori[:, 1])] + [self.pad, self.pad]
        # Remap the input points inside the resize_to x resize_to cropped box
        points_resized = (self.resize_to * points_resized * [1 / image_cropped.shape[1], 1 / image_cropped.shape[0]])

        if self.debug:
            os.makedirs("debug/", exist_ok=True)
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            plt.imshow(img)
            plt.scatter(points_ori.T[0], points_ori.T[1], c='red', s=100)
            plt.subplot(2, 1, 2)
            plt.imshow(image_resized)
            plt.scatter(points_resized.T[0], points_resized.T[1], c='red', s=100)
            plt.savefig(r"debug\PointsOutput.png")
            plt.close()

        # Set the resized image
        self.sampredictor_net.set_image(image_resized)

        # Transform the points, create labels
        input_points = points_resized.astype(int)
        input_labels = np.array([1] * len(points_resized))

        # Make prediction given points
        mask, score, logit = self.sampredictor_net.predict(point_coords=input_points,
                                                           point_labels=input_labels,
                                                           multimask_output=False)

        # If it's a good mask, else return nothing to GUI
        if score.squeeze() >= self.score_threshold:
            mask = mask.squeeze().astype(float)
        else:
            self.infoMessage.emit("Predicted mask score is too low, skipping...")
            mask = np.zeros(shape=image_resized.shape[0:2], dtype=float)

        # Resize the mask to be the same dimensions as original image
        segm_mask = helpers.crop2fullmask(mask,
                                          bbox,
                                          im_size=img.shape[:2],
                                          zero_pad=True,
                                          relax=0).astype(np.uint8)

        # Smooth mask after being resized
        kernel = np.ones((3, 3), np.uint8)
        segm_mask = cv2.morphologyEx(segm_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            plt.imshow(img)
            plt.imshow(segm_mask, alpha=0.5)
            plt.scatter(points_ori.T[0], points_ori.T[1], c='red', s=100)
            plt.subplot(2, 1, 2)
            plt.imshow(image_resized)
            plt.imshow(mask, alpha=0.5)
            plt.scatter(points_resized.T[0], points_resized.T[1], c='red', s=100)
            plt.savefig(r"debug\SegmentationOutput.png")
            plt.close()

        # TODO: move this function to blob!!!
        # SAM Masks shouldn't have multiple blobs (ideally), so only keep largest
        blobs = self.viewerplus.annotations.blobsFromMask(segm_mask,
                                                          left_map_pos,
                                                          top_map_pos,
                                                          area_mask=1000,
                                                          keep_only_largest=True)

        self.viewerplus.resetSelection()

        for blob in blobs:
            self.viewerplus.addBlob(blob, selected=True)
            self.blobInfo.emit(blob, "[TOOL][SAMPREDICTOR][BLOB-CREATED]")
        self.viewerplus.saveUndo()

        self.infoMessage.emit("Segmentation done.")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation ends.")

        QApplication.restoreOverrideCursor()
        self.resetWorkArea()

    def loadNetwork(self):

        if self.sampredictor_net is None:
            self.infoMessage.emit("Loading SAM network..")

            # Mapping between the model type, and the checkpoint file name
            sam_dict = {"vit_b": "sam_vit_b_01ec64",
                        "vit_l": "sam_vit_l_0b3195",
                        "vit_h": "sam_vit_h_4b8939"}

            # Initialization
            modelName = sam_dict[self.sam_model_type]
            models_dir = os.path.join(self.viewerplus.taglab_dir, "models")
            path = os.path.join(models_dir, modelName + '.pth')

            if not os.path.exists(path):

                # Create a box with a warning
                box = QMessageBox()
                box.setText(f"Model weights {self.sam_model_type} cannot be found in models folder.\n"
                            f"If they have not been downloaded, re-run the install script.")
                box.exec()
                # Go back to GUI without closing program

            else:
                # Set the device; users should be using a CUDA GPU, otherwise tool is slow
                device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

                # Loading the model, returning the predictor
                sam_model = sam_model_registry[self.sam_model_type](checkpoint=path)
                sam_model.to(device=device)
                self.sampredictor_net = SamPredictor(sam_model)
                self.device = device

    def resetNetwork(self):
        """
        Reset the network
        """

        torch.cuda.empty_cache()
        if self.sampredictor_net is not None:
            del self.sampredictor_net
            self.sampredictor_net = None

    def resetWorkArea(self):
        """
        Reset working area
        """
        self.work_area_bbox = [0, 0, 0, 0]
        if self.work_area_item is not None:
            self.viewerplus.scene.removeItem(self.work_area_item)
            self.work_area_item = None

    def reset(self):
        """
        Reset everything
        """
        self.resetNetwork()
        self.pick_points.reset()
        self.resetWorkArea()
