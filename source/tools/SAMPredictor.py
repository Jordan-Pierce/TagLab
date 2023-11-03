from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPen, QBrush
from source.utils import qimageToNumpyArray, cropQImage

from source.Blob import Blob
from source.tools.Tool import Tool
from source import utils

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import regionprops

import torch

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from models.dataloaders import helpers as helpers


class SAMPredictor(Tool):
    def __init__(self, viewerplus, pick_points):
        super(SAMPredictor, self).__init__(viewerplus)
        # User defined points
        self.pick_points = pick_points

        # Image is resized to
        self.resize_to = 2048
        # Padding amount
        self.pad = 0
        # Model Type (b, l, or h)
        self.sam_model_type = 'vit_b'
        # Mask score threshold
        self.score_threshold = 0.70
        # Labels for fore/background
        self.labels = []
        # For debugging
        self.debug = False

        # Mosaic dimensions
        self.width = None
        self.height = None

        # Set image
        self.image_resized = None
        self.image_cropped = None

        # SAM, CUDA or CPU
        self.sampredictor_net = None
        self.device = None

        # Drawing on GUI
        self.CROSS_LINE_WIDTH = 2
        self.work_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.cyan, 'size': 8}
        self.pos_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.green, 'size': 6}
        self.neg_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.red, 'size': 6}
        self.work_area_points = []
        self.work_area_bbox = None
        self.work_area_item = None

    def leftPressed(self, x, y, mods):
        """
        Positive points
        """
        self.loadNetwork()

        # User is still selecting work area
        if not self.sampredictor_net.is_image_set:

            # Collect points to set work area
            if len(self.pick_points.points) < 2:
                self.pick_points.addPoint(x, y, self.work_pick_style)

            # User choose a third point without pressing
            # SPACE, so reset the work area points, add
            else:
                self.pick_points.reset()
                self.pick_points.addPoint(x, y, self.work_pick_style)

        # User has already selected to work area, and now
        # is choosing positive or negative points
        else:

            # Add points
            self.pick_points.addPoint(x, y, self.pos_pick_style)
            self.labels.append(1)
            message = "[TOOL][SAMPREDICTOR] New point picked"
            self.log.emit(message)

    def rightPressed(self, x, y, mods):
        """
        Negative points
        """

        self.loadNetwork()

        if mods == Qt.ShiftModifier:

            # User is still selecting work area
            if not self.sampredictor_net.is_image_set:

                # Collect points to set work area
                if len(self.pick_points.points) < 2:
                    self.pick_points.addPoint(x, y, self.work_pick_style)

                # User choose a third point without pressing
                # SPACE, so reset the work area points, add
                else:
                    self.pick_points.reset()
                    self.pick_points.addPoint(x, y, self.work_pick_style)

            # User has already selected to work area, and now
            # is choosing positive or negative points
            else:
                # Add points
                self.pick_points.addPoint(x, y, self.neg_pick_style)
                self.labels.append(0)
                message = "[TOOL][SAMPREDICTOR] New point picked"
                self.log.emit(message)

    def apply(self):
        """
        User presses SPACE to set work area, and again later to run the model
        """

        if len(self.pick_points.points) and self.sampredictor_net.is_image_set:

            if self.points_within_workarea():
                self.segmentWithSAMPredictor()

            self.pick_points.reset()
            self.labels = []

        if len(self.pick_points.points) == 2 and not self.sampredictor_net.is_image_set:
            self.setWorkArea()

    def points_within_workarea(self):
        """
        Checks if selected points are within established work area
        """

        # Define the boundaries
        left_map_pos = self.work_area_bbox[1]
        top_map_pos = self.work_area_bbox[0]
        width_map_pos = self.work_area_bbox[2]
        height_map_pos = self.work_area_bbox[3]

        # Check if any points are outside the boundaries
        points = np.array(self.pick_points.points)
        outside_boundaries = (
                (points[:, 0] < left_map_pos) |
                (points[:, 0] > left_map_pos + width_map_pos) |
                (points[:, 1] < top_map_pos) |
                (points[:, 1] > top_map_pos + height_map_pos)
        )

        return not np.any(outside_boundaries)

    def setWorkArea(self):
        """
        Set the work area based on the location of points
        """

        points = np.array(self.pick_points.points)

        x = points[:, 0].min()
        y = points[:, 1].min()
        w = points[:, 0].max() - x
        h = points[:, 1].max() - y

        self.work_area_bbox = [round(y), round(x), round(w), round(h)]

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

        # From the current view, crop the image
        image_cropped = cropQImage(self.viewerplus.img_map, self.work_area_bbox)
        image_cropped = qimageToNumpyArray(image_cropped)

        # Resize the cropped image
        image_resized = helpers.fixed_resize(image_cropped,
                                             (self.resize_to, self.resize_to)).astype(np.uint8)

        # Retain the images
        self.image_cropped = image_cropped
        self.image_resized = image_resized

        # Set the image
        self.sampredictor_net.set_image(image_resized)
        self.pick_points.reset()

    def resizeArray(self, arr, shape, interpolation=cv2.INTER_CUBIC):
        """
        Resize array; expects 2D array.
        """
        return cv2.resize(arr.astype(float), shape, interpolation)

    def prepareForSAMPredictor(self):
        """
        Get the image based on point(s) location
        """
        points = np.asarray(self.pick_points.points).astype(int)

        left = self.work_area_bbox[1]
        top = self.work_area_bbox[0]

        # Update points to be in image_cropped coordinate space
        points_cropped = np.zeros((len(points), 2), dtype=np.int32)
        points_cropped[:, 0] = points[:, 0] - left
        points_cropped[:, 1] = points[:, 1] - top

        # Points in the resized image
        x_scale = self.image_resized.shape[1] / self.image_cropped.shape[1]
        y_scale = self.image_resized.shape[0] / self.image_cropped.shape[0]

        # New coordinates
        points_resized = np.zeros_like(points_cropped, dtype=np.float32)
        points_resized[:, 0] = points_cropped[:, 0] * x_scale
        points_resized[:, 1] = points_cropped[:, 1] * y_scale

        return points_cropped, points_resized

    def segmentWithSAMPredictor(self):

        if not self.viewerplus.img_map:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation begins..")

        # Mosaic dimensions
        self.width = self.viewerplus.img_map.size().width()
        self.height = self.viewerplus.img_map.size().height()

        # Get the work area top-left
        left_map_pos = self.work_area_bbox[1]
        top_map_pos = self.work_area_bbox[0]

        # Points in the cropped image
        points_cropped, points_resized = self.prepareForSAMPredictor()

        # Transform the points, create labels
        input_points = points_resized.astype(int)
        input_labels = np.array(self.labels)

        # Make prediction given points
        mask, score, logit = self.sampredictor_net.predict(point_coords=input_points,
                                                           point_labels=input_labels,
                                                           multimask_output=False)

        # If mask score is too low, just return early
        if score.squeeze() < self.score_threshold:
            self.infoMessage.emit("Predicted mask score is too low, skipping...")
        else:
            # Get the mask as a float
            mask_resized = mask.squeeze()
            # Fill in while still small
            mask_resized = ndi.binary_fill_holes(mask_resized).astype(float)

            # Region contain masked object
            indices = np.argwhere(mask_resized)

            # Calculate the x, y, width, and height
            x = indices[:, 1].min()
            y = indices[:, 0].min()
            w = indices[:, 1].max() - x + 1
            h = indices[:, 0].max() - y + 1
            bbox = np.array([x, y, w, h])

            # Resize mask back to cropped size
            target_shape = (self.image_cropped.shape[:2][::-1])
            mask_cropped = self.resizeArray(mask_resized, target_shape, cv2.INTER_CUBIC)
            mask_cropped = mask_cropped.astype(np.uint8)

            if self.debug:
                os.makedirs("debug", exist_ok=True)
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 1, 1)
                plt.imshow(self.image_cropped)
                plt.imshow(mask_cropped, alpha=0.5)
                plt.scatter(points_cropped.T[0], points_cropped.T[1], c='red', s=100)
                plt.subplot(2, 1, 2)
                plt.imshow(self.image_resized)
                plt.imshow(mask_resized, alpha=0.5)
                plt.scatter(points_resized.T[0], points_resized.T[1], c='red', s=100)
                plt.savefig(r"debug\SegmentationOutput.png")
                plt.close()

            # Create a blob manually using provided information
            blob = self.createBlob(mask_resized, mask_cropped, bbox, left_map_pos, top_map_pos)

            self.viewerplus.resetSelection()

            if blob:
                self.viewerplus.addBlob(blob, selected=True)
                self.blobInfo.emit(blob, "[TOOL][SAMPREDICTOR][BLOB-CREATED]")
            self.viewerplus.saveUndo()

        self.infoMessage.emit("Segmentation done.")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation ends.")

        QApplication.restoreOverrideCursor()

    def createBlob(self, mask_src, mask_dst, bbox_src, left_map_pos, top_map_pos):
        """
        Create a blob manually given the generated mask
        """

        # Bbox of the area of interest before scaled
        x1_src, y1_src, w_src, h_src = bbox_src

        # Calculate scale
        x_scale = mask_dst.shape[1] / mask_src.shape[1]
        y_scale = mask_dst.shape[0] / mask_src.shape[0]

        # New coordinates
        x1_dst = x1_src * x_scale
        y1_dst = y1_src * y_scale
        w_dst = w_src * x_scale
        h_dst = h_src * y_scale

        # Bbox of the area of interest after scaled
        bbox_dst = (x1_dst, y1_dst, (x1_dst + w_dst), (y1_dst + h_dst))

        try:
            # Create region manually since information is available;
            # It's also much faster than using scikit measure

            # Inside a try block because scikit complains, but still
            # takes the values anyway
            region = sorted(regionprops(mask_dst), key=lambda r: r.area, reverse=True)[0]
            region.label = 1
            region.bbox = bbox_dst
            region.area = np.sum(mask_dst)
            region.centroid = np.mean(np.argwhere(mask_dst), axis=0)
        except:
            pass

        blob_id = self.viewerplus.annotations.getFreeId()
        blob = Blob(region, left_map_pos, top_map_pos, blob_id)

        return blob

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
        self.image_resized = None
        self.image_cropped = None
        self.work_area_points = []
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
        self.labels = []
        self.resetWorkArea()
