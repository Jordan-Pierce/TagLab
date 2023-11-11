from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPen, QBrush
from source.utils import qimageToNumpyArray, cropQImage

from source.Blob import Blob
from source.tools.Tool import Tool

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import regionprops

import torch
import torchvision

from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from segment_anything.utils.amg import build_all_layer_point_grids

from models.dataloaders import helpers as helpers


class SAMGenerator(Tool):
    def __init__(self, viewerplus, pick_points):
        super(SAMGenerator, self).__init__(viewerplus)
        # User defined points
        self.pick_points = pick_points

        # Image is resized to
        self.resize_to = 1024
        # Model Type (b, l, or h)
        self.sam_model_type = 'vit_b'
        # Mask score threshold
        self.score_threshold = 0.80
        # IoU score threshold
        self.iou_threshold = 0.3
        # Number of points
        self.num_points = 4
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
        self.work_area_bbox = None
        self.work_area_item = None
        self.work_points = []

        self.CROSS_LINE_WIDTH = 2
        self.work_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.cyan, 'size': 8}
        self.pos_pick_style = {'width': self.CROSS_LINE_WIDTH, 'color': Qt.green, 'size': 4}

    def leftPressed(self, x, y, mods):
        """

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

        # User has already selected a work area, and now
        # is choosing to increase or decrease the amount of points
        else:
            self.setWorkPoints(1)

    def rightPressed(self, x, y, mods):
        """

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

        # User has already selected a work area, and now
        # is choosing to increase or decrease the amount of points
        else:
            self.setWorkPoints(-1)

    def apply(self):
        """
        User presses SPACE to set work area, and again later to run the model
        """

        # User has already selected work area, and pressed SPACE
        if self.sampredictor_net.is_image_set:
            self.segmentWithSAMPredictor()

        # User has finished creating working area, saving work area
        if len(self.pick_points.points) == 2 and not self.sampredictor_net.is_image_set:
            self.setWorkArea()
            self.setWorkPoints()

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

        # Resize the cropped QImage
        self.image_resized = helpers.fixed_resize(self.image_cropped,
                                                  (self.resize_to, self.resize_to)).astype(np.uint8)

        # Prepare via CUDA
        x = self.prepareImage(self.image_resized)

        # Do use torch here, as it's fast as hell
        self.sampredictor_net.set_torch_image(x, self.image_resized.shape[0:2])
        self.sampredictor_net.set_image(self.image_resized)
        self.pick_points.reset()

    def prepareImage(self, arr):
        """

        """
        shape = (self.resize_to, self.resize_to)
        trans = torchvision.transforms.Compose([torchvision.transforms.Resize(shape, antialias=True)])
        image = torch.as_tensor(arr).cuda()
        transformed_image = trans(image.permute(2, 0, 1)).unsqueeze(0)

        return transformed_image

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

    def preparePoints(self):
        """
        Get grid of points
        """

        points_resized = build_all_layer_point_grids(self.num_points, 0, 1)[0] * self.resize_to

        return points_resized

    def resizeArray(self, arr, shape, interpolation=cv2.INTER_CUBIC):
        """
        Resize array; expects 2D array.
        """
        return cv2.resize(arr.astype(float), shape, interpolation)

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
        points_resized = self.preparePoints()
        # Labels for the points
        point_labels = np.array([1] * len(points_resized))

        # Convert to torch, cuda
        input_labels = torch.tensor(point_labels).to(self.device).unsqueeze(1)

        input_points = torch.as_tensor(points_resized.astype(int), dtype=torch.int64).to(self.device).unsqueeze(1)
        transformed_points = self.sampredictor_net.transform.apply_coords_torch(input_points,
                                                                                self.image_resized.shape[:2])

        try:
            # Make prediction given points
            masks, scores, logits = self.sampredictor_net.predict_torch(point_coords=transformed_points,
                                                                        point_labels=input_labels,
                                                                        multimask_output=False)
        except:
            # Create a box with a warning
            box = QMessageBox()
            box.setText(f"You selected more points than your GPU can handle!")
            box.exec()
            self.reset()
            return False

        # Squeeze
        masks = masks.squeeze()
        scores = scores.squeeze()

        # Filter the masks to save time
        masks = masks[scores > self.score_threshold]
        masks = self.removeSimilarMasks(masks, self.iou_threshold)

        for idx, generated_output in enumerate(masks):

            # Get the current mask
            mask_resized = masks[idx]
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
            mask_cropped = self.smoothVertices(mask_cropped).astype(np.uint8)

            if self.debug:
                os.makedirs("debug", exist_ok=True)
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 1, 1)
                plt.imshow(self.image_cropped)
                plt.imshow(mask_cropped, alpha=0.5)
                plt.subplot(2, 1, 2)
                plt.imshow(self.image_resized)
                plt.imshow(mask_resized, alpha=0.5)
                plt.scatter(points_resized.T[0], points_resized.T[1], c='red', s=100)
                plt.savefig(r"debug\SegmentationOutput.png")
                plt.close()

            # Create a blob manually using provided information
            blob = self.createBlob(mask_resized, mask_cropped, bbox, left_map_pos, top_map_pos)

            if blob:
                self.viewerplus.addBlob(blob, selected=True)
                self.blobInfo.emit(blob, "[TOOL][SAMGENERATOR][BLOB-CREATED]")
            self.viewerplus.saveUndo()

        self.infoMessage.emit("Segmentation done.")
        self.log.emit("[TOOL][SAMPREDICTOR] Segmentation ends.")

        self.reset()
        QApplication.restoreOverrideCursor()

    def binaryMaskIOU(self, mask1, mask2):
        """

        """
        mask1_area = torch.sum(mask1)
        mask2_area = torch.sum(mask2)
        intersection = torch.sum(mask1 * mask2)
        iou = intersection / (mask1_area + mask2_area - intersection)

        return iou.item()

    def removeSimilarMasks(self, mask_list, iou_threshold):
        """

        """

        # Create an empty list to store the unique masks
        unique_masks = []

        # Iterate through the input mask list
        for mask in mask_list:
            # Flag to keep track if the mask is similar to any existing unique mask
            is_similar = False

            # Iterate through the unique masks to compare with the current mask
            for unique_mask in unique_masks:
                iou = self.binaryMaskIOU(mask, unique_mask)
                if iou >= iou_threshold:
                    is_similar = True
                    break

            # If the mask is not similar to any existing unique mask, add it to the list
            if not is_similar:
                unique_masks.append(mask)

        # Convert the unique masks back to NumPy arrays for the output
        unique_masks = [mask.cpu().numpy().astype(bool) for mask in unique_masks]

        return unique_masks

    def smoothVertices(self, arr):

        # Find the contours in the binary mask
        contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Choose the largest contour if there are multiple
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify the contour with a specified tolerance
        epsilon = 0.0001 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Create a new binary mask with the smoothed contour
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        cv2.fillPoly(new_arr, [simplified_contour], 1)

        return new_arr

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
            eps = 3

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
        self.num_points = 4
        self.resetWorkArea()
