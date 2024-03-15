# TagLab
# A semi-automatic segmentation tool
#
# Copyright(C) 2019
# Visual Computing Lab
# ISTI - Italian National Research Council
# All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License (http://www.gnu.org/licenses/gpl.txt)
# for more details.

import os
import sys
import glob
from io import TextIOBase

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QMessageBox, QTextEdit
from PyQt5.QtWidgets import QSizePolicy, QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout

from source.tools.CoralNetToolbox.API import submit_jobs
from source.tools.CoralNetToolbox.Upload import upload_images
from source.tools.CoralNetToolbox.Common import get_now, IMG_FORMATS
from source.tools.CoralNetToolbox.Browser import login, authenticate, check_for_browsers, get_token


class ConsoleOutput(TextIOBase):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def write(self, string):
        self.widget.append(string.strip())
        QApplication.processEvents()


class ConsoleWidget(QTextEdit):
    def __init__(self, parent=None):
        super(ConsoleWidget, self).__init__(parent)
        self.setReadOnly(True)
        self._console_output = ConsoleOutput(self)
        sys.stdout = self._console_output
        sys.stderr = self._console_output

    def clearConsole(self):
        self.clear()


class CoralNetToolboxWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super(CoralNetToolboxWidget, self).__init__(parent)

        # Parameters
        self.username = ""
        self.password = ""
        self.source_id_1 = ""
        self.source_id_2 = ""
        self.output_folder = ""
        self.tiles_folder = ""
        self.annotations_file = ""
        self.predictions_file = ""

        # Driver
        self.driver = None

        # Tile size
        self.tile_size = 1024

        # --------------------
        # The window settings
        # --------------------
        self.setWindowTitle("CoralNet Toolbox")

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        # Set size policy to allow resizing
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(1000, 600)

        self.setStyleSheet("background-color: rgba(60,60,65,100); color: white")

        # -----------------------
        # CoralNet Parameters
        # -----------------------

        # Username
        layoutUsername = QHBoxLayout()
        layoutUsername.setAlignment(Qt.AlignLeft)
        self.lblUsername = QLabel("Username: ")
        self.lblUsername.setFixedWidth(130)
        self.lblUsername.setMinimumWidth(130)
        self.editUsername = QLineEdit("")
        self.editUsername.setPlaceholderText("CoralNet Username")
        self.editUsername.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnUsername = QPushButton("...")
        self.btnUsername.setFixedWidth(20)
        self.btnUsername.setMinimumWidth(20)
        self.btnUsername.clicked.connect(self.getCoralNetUsername)

        layoutUsername.addWidget(self.lblUsername)
        layoutUsername.addWidget(self.editUsername)
        layoutUsername.addWidget(self.btnUsername)

        # Password
        layoutPassword = QHBoxLayout()
        layoutPassword.setAlignment(Qt.AlignLeft)
        self.lblPassword = QLabel("Password: ")
        self.lblPassword.setFixedWidth(130)
        self.lblPassword.setMinimumWidth(130)
        self.editPassword = QLineEdit("")
        self.editPassword.setEchoMode(QLineEdit.Password)
        self.editPassword.setPlaceholderText("CoralNet Password")
        self.editPassword.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnPassword = QPushButton("...")
        self.btnPassword.setFixedWidth(20)
        self.btnPassword.setMinimumWidth(20)
        self.btnPassword.clicked.connect(self.getCoralNetPassword)

        layoutPassword.addWidget(self.lblPassword)
        layoutPassword.addWidget(self.editPassword)
        layoutPassword.addWidget(self.btnPassword)

        # Authorization button
        # TODO add button to call self.coralnetAuthenticate

        # Source ID 1 (images)
        layoutSourceID1 = QHBoxLayout()
        layoutSourceID1.setAlignment(Qt.AlignLeft)
        self.lblSourceID1 = QLabel("Source ID 1: ")
        self.lblSourceID1.setFixedWidth(130)
        self.lblSourceID1.setMinimumWidth(130)
        self.editSourceID1 = QLineEdit("")
        self.editSourceID1.setPlaceholderText("Source ID for storing images")
        self.editSourceID1.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        layoutSourceID1.addWidget(self.lblSourceID1)
        layoutSourceID1.addWidget(self.editSourceID1)

        # Source ID 2 (model)
        layoutSourceID2 = QHBoxLayout()
        layoutSourceID2.setAlignment(Qt.AlignLeft)
        self.lblSourceID2 = QLabel("Source ID 2: ")
        self.lblSourceID2.setFixedWidth(130)
        self.lblSourceID2.setMinimumWidth(130)
        self.editSourceID2 = QLineEdit("")
        self.editSourceID2.setPlaceholderText("Source ID for desired model")
        self.editSourceID2.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        layoutSourceID2.addWidget(self.lblSourceID2)
        layoutSourceID2.addWidget(self.editSourceID2)

        # Output Folder
        layoutOutputFolder = QHBoxLayout()
        layoutOutputFolder.setAlignment(Qt.AlignLeft)
        self.lblOutputFolder = QLabel("Output Folder: ")
        self.lblOutputFolder.setFixedWidth(130)
        self.lblOutputFolder.setMinimumWidth(130)
        self.editOutputFolder = QLineEdit("")
        self.editOutputFolder.setPlaceholderText("")
        # TODO set placeholder text to data folder
        self.editOutputFolder.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnOutputFolder = QPushButton("...")
        self.btnOutputFolder.setFixedWidth(20)
        self.btnOutputFolder.setMinimumWidth(20)
        self.btnOutputFolder.clicked.connect(self.chooseOutputFolder)

        layoutOutputFolder.addWidget(self.lblOutputFolder)
        layoutOutputFolder.addWidget(self.editOutputFolder)
        layoutOutputFolder.addWidget(self.btnOutputFolder)

        # Tile size (combobox, power of 2)
        # TODO Add tile size limit (CoralNet allows 8k x 8k)

        # Delete temporary data (check box)
        # TODO boolean to delete temporary data in self.output_dir (make sure it's the right folder)

        # -----------------------
        # Source List Widget
        # -----------------------

        # -----------------------
        # Label Mapping Widget
        # -----------------------

        # -----------------------
        # Console Widget
        # -----------------------
        self.console_widget = ConsoleWidget()
        self.console_widget.setFixedHeight(400)

        # -----------------------
        # Buttons
        # -----------------------
        layoutButtons = QHBoxLayout()
        layoutButtons.setAlignment(Qt.AlignRight)
        layoutButtons.addStretch()

        self.btnApply = QPushButton("Apply")
        self.btnApply.clicked.connect(self.apply)
        self.btnExit = QPushButton("Exit")
        self.btnExit.clicked.connect(self.close)

        layoutButtons.addWidget(self.btnApply)
        layoutButtons.addWidget(self.btnExit)

        # -----------------------
        # Final Layout order
        # -----------------------
        layoutInfo = QVBoxLayout()

        # Add your existing layout elements
        layoutInfo.addLayout(layoutUsername)
        layoutInfo.addLayout(layoutPassword)
        layoutInfo.addLayout(layoutSourceID1)
        layoutInfo.addLayout(layoutSourceID2)
        layoutInfo.addLayout(layoutOutputFolder)
        layoutInfo.addWidget(self.console_widget)
        layoutInfo.addLayout(layoutButtons)

        layoutV = QVBoxLayout()
        layoutV.addLayout(layoutInfo)

        self.setLayout(layoutV)

    @pyqtSlot()
    def getCoralNetUsername(self):
        """

        """
        coralnetUsername = os.getenv('CORALNET_USERNAME')
        if coralnetUsername:
            self.editUsername.setText(coralnetUsername)

    @pyqtSlot()
    def getCoralNetPassword(self):
        """

        """
        coralnetPassword = os.getenv('CORALNET_PASSWORD')
        if coralnetPassword:
            self.editPassword.setText(coralnetPassword)

    @pyqtSlot()
    def chooseOutputFolder(self):
        """

        """
        folder_name = QFileDialog.getExistingDirectory(self, "Choose a Folder as the root working directory", "")
        if folder_name:
            self.editOutputFolder.setText(folder_name)

    @pyqtSlot()
    def apply(self):
        """

        """
        # Good or Bad box
        msgBox = QMessageBox()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.username = self.editUsername.text()
        self.password = self.editPassword.text()
        self.source_id_1 = int(self.editSourceID1.text())
        self.source_id_2 = int(self.editSourceID2.text())
        self.output_folder = self.editOutputFolder.text()
        self.output_folder = f"{self.output_folder}/{get_now()}"

        # TODO this will be after the authenticate button
        self.coralnetAuthenticate()
        self.console_widget.clearConsole()

        try:
            # Export point annotations and tiles from
            # active viewer based on user defined area
            self.taglabExport()

            # Use upload function to upload just
            # the images in the tiles folder
            self.coralnetUpload()

            # Use api function and point annotations
            # to get predictions for uploaded tiles
            self.coralnetAPI()

            # Import predictions back to TagLab
            self.taglabImport()

            QApplication.restoreOverrideCursor()
            msgBox.setText(f"Imported predictions successfully")
            msgBox.exec()

            # Close the widget
            self.close()

        except Exception as e:
            self.console_widget.clearConsole()
            QApplication.restoreOverrideCursor()
            msgBox.setText(f"{e}")
            msgBox.exec()

    @pyqtSlot()
    def coralnetAuthenticate(self):
        """
        Authenticates the username and password before starting the process.
        """
        # Good or Bad box
        msgBox = QMessageBox()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Check that the credentials are correct
            authenticate(self.username, self.password)

            # Cache the Username and Password as local variables
            os.environ["CORALNET_USERNAME"] = self.username
            os.environ["CORALNET_PASSWORD"] = self.password

            # Use the username and password to get the token
            token, headers = get_token(self.username, self.password)

            # Check for Browsers
            self.driver = check_for_browsers(headless=True)
            # Store the credentials in the driver
            self.driver.capabilities['credentials'] = {
                'username': self.username,
                'password': self.password,
                'headers': headers,
                'token': token
            }
            # Log in to CoralNet
            self.driver, _ = login(self.driver)

            # Autofill Source ID lists
            # TODO

        except Exception as e:
            msgBox.setText(f"Could not authenticate with provided credentials. {e}")
            msgBox.exec()

        QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def taglabUpdateSources(self):
        """

        """
        pass

    def taglabExport(self):
        """
        Exports the points and tiles from the user specified area in the output directory;
        returns the path to the CSV file.
        """
        # Get the channel, annotations and working area from project
        channel = self.parent().activeviewer.image.getRGBChannel()
        annotations = self.parent().activeviewer.annotations
        working_area = self.parent().project.working_area

        # Export the data
        output_dir, csv_file = self.parent().activeviewer.annotations.exportCoralNetData(self.output_folder,
                                                                                         channel,
                                                                                         annotations,
                                                                                         working_area,
                                                                                         self.tile_size)
        if os.path.exists(csv_file):
            self.output_folder = f"{output_dir}"
            self.tiles_folder = f"{output_dir}/tiles"
            self.annotations_file = csv_file
            print("NOTE: Data exported successfully")
        else:
            raise Exception("TagLab annotations could not be exported")

    def coralnetUpload(self):
        """

        """
        try:
            # Get all the tiles from the tiles folder
            images = os.path.abspath(self.tiles_folder)
            images = glob.glob(images + "/*.*")
            images = [i for i in images if i.split('.')[-1].lower() in IMG_FORMATS]

            # Check if there are images to upload
            if len(images) > 0:
                print(f"NOTE: Found {len(images)} images to upload")
            else:
                raise Exception(f"No valid images found in {self.tiles_folder}")

            # Run the upload function
            self.driver, _ = upload_images(self.driver, self.source_id_1, images, prefix="")
            print("NOTE: Data uploaded successfully")

        except Exception as e:
            raise Exception(f"CoralNet upload failed. {e}")

    def coralnetAPI(self):
        """

        """
        try:
            # Make sure the file exists
            if os.path.exists(self.annotations_file):
                # Read it in
                points = pd.read_csv(self.annotations_file, index_col=0)
            else:
                raise Exception(f"{self.annotations_file} does not exist")

            # Check to see if the csv file has the expected columns
            assert 'Name' in points.columns, "'Name' field not found in file"
            assert 'Row' in points.columns, "'Row' field not found in file"
            assert 'Column' in points.columns, "'Column' field not found in file"
            assert len(points) > 0, "No points found in file"

            # Convert list of names to a list
            images_w_points = points['Name'].to_list()

            # Run the CoralNet API function
            self.driver, _, self.predictions_file = submit_jobs(self.driver,
                                                                self.source_id_1,
                                                                self.source_id_2,
                                                                "",
                                                                images_w_points,
                                                                points,
                                                                self.output_folder)
            # Check that the file was created
            if os.path.exists(self.predictions_file):
                print("NOTE: Predictions made successfully")
            else:
                raise Exception("Predictions file was not created")

        except Exception as e:
            raise Exception(f"CoralNet API failed. {e}")

    def taglabImport(self):
        """

        """
        try:
            # Get the channel for the orthomosaic
            channel = self.parent().activeviewer.image.getRGBChannel()
            self.parent().activeviewer.annotations.importCoralNetCSVAnn(self.predictions_file, channel)
            print("NOTE: Predictions imported successfully")
        except Exception as e:
            raise Exception("TagLab annotations could not be imported")

    def closeEvent(self, event):
        """

        """
        self.driver = None
        sys.stdout = sys.__stdout__
        super().closeEvent(event)