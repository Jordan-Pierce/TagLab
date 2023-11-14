from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPen, QBrush
from source.utils import qimageToNumpyArray, cropQImage

from source.Blob import Blob
from source.tools.Tool import Tool

import os
import cv2
import numpy as np
from skimage.measure import regionprops

import torch

from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids

from models.dataloaders import helpers as helpers


class SAMGenerator(Tool):
    def __init__(self, viewerplus, pick_points):
        super(SAMGenerator, self).__init__(viewerplus)
        # User defined points
        self.pick_points = pick_points

        # Image is resized to
        self.longest_side = 1024
        # Resize dimensions
        self.resize_width = None
        self.resize_height = None
        # Model Type (b, l, or h)
        self.sam_model_type = 'vit_b'
        # Number of points
        self.num_points = 32
        # For debugging
        self.debug = False

        # Mosaic dimensions
        self.width = None
        self.height = None

        # Set image
        self.image_resized = None
        self.image_cropped = None

        # SAM, CUDA or CPU
        self.samgenerator_net = None
        self.device = None

        # Drawing on GUI
        self.work_area_set = False
        self.work_area_bbox = None
        self.work_area_item = None
        self.work_points = []

        self.CROSS_LINE_WIDTH = 2
        self.work_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.cyan, 'size': 8}
        self.pos_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.green, 'size': 4}

    def leftPressed(self, x, y, mods):
        """

        """
        # User is still selecting work area
        if not self.work_area_set:

            # Collect points to set work area
            if len(self.pick_points.points) < 2:
                self.pick_points.addPoint(x, y, self.work_pick_style)

            # User choose a third point without pressing
            # SPACE, so reset the work area points, add
            else:
                self.pick_points.reset()
                self.pick_points.addPoint(x, y, self.work_pick_style)

        # User has already selected a work area, and now
        # is choosing to increase or decrease the amount of points
        else:
            self.setWorkPoints(5)

    def rightPressed(self, x, y, mods):
        """

        """
        # User is still selecting work area
        if not self.work_area_set:

            # Collect points to set work area
            if len(self.pick_points.points) < 2:
                self.pick_points.addPoint(x, y, self.work_pick_style)

            # User choose a third point without pressing
            # SPACE, so reset the work area points, add
            else:
                self.pick_points.reset()
                self.pick_points.addPoint(x, y, self.work_pick_style)

        # User has already selected a work area, and now
        # is choosing to increase or decrease the amount of points
        else:
            self.setWorkPoints(-5)

    def apply(self):
        """
        User presses SPACE to set work area, and again later to run the model
        """

        # User has chosen the current view as the working area, saving work area
        if len(self.pick_points.points) == 0 and not self.work_area_set:
            self.getExtent()
            self.setWorkArea()
            self.setWorkPoints()

        # User has finished creating working area, saving work area
        elif len(self.pick_points.points) == 2 and not self.work_area_set:
            self.setWorkArea()
            self.setWorkPoints()

        # User has already selected work area, and pressed SPACE
        elif self.work_area_set:
            self.loadNetwork()
            self.segmentWithSAMGenerator()

    def getExtent(self):
        """

        """

        # Mosaic dimensions
        self.width = self.viewerplus.img_map.size().width()
        self.height = self.viewerplus.img_map.size().height()

        # Current extent
        rect_map = self.viewerplus.viewportToScene()

        top = round(rect_map.top())
        left = round(rect_map.left())
        width = round(rect_map.width())
        height = round(rect_map.height())
        bottom = top + height
        right = left + width

        # If the current extent includes areas outside the
        # mosaic, reduce it to be only the mosaic
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > self.height:
            bottom = self.height
        if right > self.width:
            right = self.width

        self.pick_points.addPoint(left, top, self.work_pick_style)
        self.pick_points.addPoint(left, bottom, self.work_pick_style)
        self.pick_points.addPoint(right, bottom, self.work_pick_style)
        self.pick_points.addPoint(right, top, self.work_pick_style)

    def setAspectRatio(self):
        """

        """

        # Calculate the original width and height
        original_height, original_width = self.image_cropped.shape[0:2]

        # Calculate the new height to preserve the aspect ratio
        new_height = int((self.longest_side / original_width) * original_height)

        self.resize_width = self.longest_side
        self.resize_height = new_height

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
        self.image_cropped = qimageToNumpyArray(image_cropped)

        # Set the aspect ratio of the cropped image
        self.setAspectRatio()

        # Resize the cropped QImage
        self.image_resized = helpers.fixed_resize(self.image_cropped,
                                                  (self.resize_width, self.resize_height)).astype(np.uint8)

        # Reset the points
        self.pick_points.reset()
        self.work_area_set = True

    def setWorkPoints(self, delta=0):
        """

        """
        # Reset the current points shown
        self.pick_points.reset()

        # Change the number of points to display
        self.num_points += delta

        # Get the updated number of points
        x_pts, y_pts = build_all_layer_point_grids(self.num_points, 0, 1)[0].T

        left = self.work_area_bbox[1]
        top = self.work_area_bbox[0]

        # Change coordinates to match viewer
        x_pts = (x_pts * self.image_cropped.shape[1]) + left
        y_pts = (y_pts * self.image_cropped.shape[0]) + top

        # Add all of them to the list
        for x, y in list(zip(x_pts, y_pts)):
            self.pick_points.addPoint(x, y, self.pos_pick_style)

    def resizeArray(self, arr, shape, interpolation=cv2.INTER_CUBIC):
        """
        Resize array; expects 2D array.
        """
        return cv2.resize(arr.astype(float), shape, interpolation)

    def segmentWithSAMGenerator(self):

        if not self.viewerplus.img_map:
            return

        self.infoMessage.emit("Segmentation is ongoing..")
        self.log.emit("[TOOL][SAMGENERATOR] Segmentation begins..")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Mosaic dimensions
        self.width = self.viewerplus.img_map.size().width()
        self.height = self.viewerplus.img_map.size().height()

        # Get the work area top-left
        left_map_pos = self.work_area_bbox[1]
        top_map_pos = self.work_area_bbox[0]

        # Generate masks
        generated_outputs = self.samgenerator_net.generate(self.image_resized)

        for idx, generated_output in enumerate(generated_outputs):

            # Extract the generated output
            area = generated_output['area']
            mask_resized = generated_output['segmentation']
            # Maybe use these to filter masks?
            iou_score = generated_output['predicted_iou']
            stable_score = generated_output['stability_score']
            # Image dimensions
            crop_box = generated_output['crop_box']
            # Region contain masked object
            bbox = generated_output['bbox']

            # Resize mask back to cropped size
            target_shape = (self.image_cropped.shape[:2][::-1])
            mask_cropped = self.resizeArray(mask_resized, target_shape, cv2.INTER_CUBIC)
            mask_cropped = mask_cropped.astype(np.uint8)

            # Create a blob manually using provided information
            blob = self.createBlob(mask_resized, mask_cropped, bbox, left_map_pos, top_map_pos)

            if blob:
                self.viewerplus.addBlob(blob, selected=True)
                self.blobInfo.emit(blob, "[TOOL][SAMGENERATOR][BLOB-CREATED]")
            self.viewerplus.saveUndo()

        QApplication.restoreOverrideCursor()
        self.infoMessage.emit("Segmentation done.")
        self.log.emit("[TOOL][SAMGENERATOR] Segmentation ends.")
        self.resetWorkArea()

    def createBlob(self, mask_src, mask_dst, bbox_src, left_map_pos, top_map_pos, omit_border_masks=True):
        """
        Create a blob manually given the generated mask
        """

        # Bbox of the area of interest before scaled
        x1_src, y1_src, w_src, h_src = bbox_src
        x2_src, y2_src = x1_src + w_src, y1_src + h_src

        # Calculate scale
        x_scale = mask_dst.shape[1] / mask_src.shape[1]
        y_scale = mask_dst.shape[0] / mask_src.shape[0]

        # New coordinates
        x1_dst = x1_src * x_scale
        y1_dst = y1_src * y_scale
        w_dst = w_src * x_scale
        h_dst = h_src * y_scale

        x2_dst = x1_dst + w_dst
        y2_dst = y1_dst + h_dst

        # Bbox of the area of interest after scaled
        bbox_dst = (x1_dst, y1_dst, (x1_dst + w_dst), (y1_dst + h_dst))

        if omit_border_masks:

            # ********************************************************
            # Remove masks that form at the boundaries?
            # May remove this if users prefer to keep those and clean
            # them manually afterwards, but it appears to be more work
            # ********************************************************
            eps = 5

            # Is the mask along the:
            min_mosaic = True
            max_mosaic = True
            min_image = True
            max_image = True

            # If below the minimum boundaries of mosaic, that's not okay
            if np.all(np.array([x1_dst, y1_dst, x2_dst, y2_dst]) >= 0):
                min_mosaic = False

            # If along the maximum boundaries of mosaic, that's okay
            if x2_dst <= self.width or y2_dst <= self.height:
                max_mosaic = False

            # If along any of the minimum boundaries of resized image, that's not okay
            if np.all(np.array([x1_src, y1_src]) >= 0 + eps):
                min_image = False

            # If along any of the minimum boundaries of resized image, that's not okay
            if x2_src <= mask_src.shape[1] - eps and y2_src <= mask_src.shape[0] - eps:
                max_image = False

            # If any of the above conditions are true, don't keep mask
            if np.any(np.array([min_mosaic, max_mosaic, min_image, max_image])):
                return None

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

        if self.samgenerator_net is None:
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
                self.samgenerator_net = SamAutomaticMaskGenerator(sam_model,
                                                                  points_per_side=self.num_points,
                                                                  points_per_batch=128)
                self.device = device

    def resetNetwork(self):
        """
        Reset the network
        """

        torch.cuda.empty_cache()
        if self.samgenerator_net is not None:
            del self.samgenerator_net
            self.samgenerator_net = None

    def resetWorkArea(self):
        """
        Reset working area
        """
        self.num_points = 32
        self.pick_points.reset()
        self.image_resized = None
        self.image_cropped = None
        self.work_area_set = False
        self.work_area_bbox = [0, 0, 0, 0]
        if self.work_area_item is not None:
            self.viewerplus.scene.removeItem(self.work_area_item)
            self.work_area_item = None

    def reset(self):
        """
        Reset everything
        """
        self.resetNetwork()
        self.resetWorkArea()
