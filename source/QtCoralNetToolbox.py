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
import json
from io import TextIOBase

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QMessageBox, QTextEdit
from PyQt5.QtWidgets import QSizePolicy, QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QScrollArea, QHeaderView, QComboBox

from source.tools.CoralNetToolbox.API import submit_jobs
from source.tools.CoralNetToolbox.Upload import upload_images
from source.tools.CoralNetToolbox.Common import get_now, IMG_FORMATS
from source.tools.CoralNetToolbox.Browser import login, authenticate, check_for_browsers, get_token
from source.tools.CoralNetToolbox.Download import download_coralnet_sources, download_labelset


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

        # Default output folder
        self.temp_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.temp_folder = f"{self.temp_folder}\\temp\\Points"

        # Parameters
        self.username = ""
        self.password = ""
        self.source_id_1 = ""
        self.source_id_2 = ""
        self.output_folder = ""
        self.tiles_folder = ""
        self.annotations_file = ""
        self.predictions_file = ""

        # CoralNet dataframes
        self.source_list = None
        self.labelset = None

        # TagLab labels
        self.taglab_labels = sorted(list(self.parent().project.labels.keys()))

        # Driver
        self.driver = None

        # Tile size
        self.tile_size = 1024

        # --------------------
        # The window settings
        # --------------------

        self.setWindowTitle("CoralNet Toolbox")

        # Set window flags for fullscreen, minimize, and exit only
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint)

        # Set window size policy to fixed size to disable resizing
        self.setFixedSize(1800, 800)

        self.setStyleSheet("background-color: rgba(60,60,65,100); color: white")

        # -------------------------
        # Source List (Left Panel)
        # -------------------------

        # Header Label
        header_label = QLabel("Source List")
        header_label.setStyleSheet("font-weight: bold; font-size: 24px; color: white;")
        header_label.setAlignment(Qt.AlignCenter)

        # Scroll Area for Source List Widget
        self.source_list_scroll_area = QScrollArea()
        self.source_list_scroll_area.setWidgetResizable(True)
        self.source_list_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.source_list_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Source List Table Widget
        self.source_list_table = QTableWidget()
        self.source_list_table.setColumnCount(3)
        self.source_list_table.setHorizontalHeaderLabels(["Source ID", "Name", "URL"])

        # Set header stylesheet
        header_style = "::section { background-color: rgba(60, 60, 65, 100); color: white; }"
        self.source_list_table.horizontalHeader().setStyleSheet(header_style)
        self.source_list_table.verticalHeader().setStyleSheet(header_style)

        # Set the last column to stretch
        self.source_list_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

        # Set items in table to be non-editable
        self.source_list_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.source_list_scroll_area.setWidget(self.source_list_table)

        # Source List Widget Layout
        self.layoutLeftPanel = QVBoxLayout()

        # Add header and informational text to the layout
        self.layoutLeftPanel.addWidget(header_label)
        self.layoutLeftPanel.addWidget(self.source_list_scroll_area)

        # -----------------------------------
        # CoralNet Parameters (Middle Panel)
        # -----------------------------------

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
        layoutAuthorizationBtn = QHBoxLayout()
        layoutAuthorizationBtn.setAlignment(Qt.AlignLeft)
        button_stylesheet = "QPushButton { background-color: rgb(150, 150, 150); }"
        self.btnAuthorization = QPushButton("Authenticate")
        self.btnAuthorization.clicked.connect(self.coralnetAuthenticate)
        self.btnAuthorization.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btnAuthorization.setStyleSheet(button_stylesheet)
        self.btnAuthorization.setMinimumWidth(500)
        layoutAuthorizationBtn.addWidget(self.btnAuthorization)

        # Source ID 1 (images)
        layoutSourceID1 = QHBoxLayout()
        layoutSourceID1.setAlignment(Qt.AlignLeft)
        self.lblSourceID1 = QLabel("Source ID 1: ")
        self.lblSourceID1.setFixedWidth(130)
        self.lblSourceID1.setMinimumWidth(130)
        self.editSourceID1 = QLineEdit("")
        self.editSourceID1.setReadOnly(True)
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
        self.editSourceID2.setReadOnly(True)
        self.editSourceID2.setPlaceholderText("Source ID for desired model")
        self.editSourceID2.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")

        layoutSourceID2.addWidget(self.lblSourceID2)
        layoutSourceID2.addWidget(self.editSourceID2)

        # Set Sources button
        layoutSetSourcesBtn = QHBoxLayout()
        layoutSetSourcesBtn.setAlignment(Qt.AlignLeft)
        button_stylesheet = "QPushButton { background-color: rgb(150, 150, 150); }"
        self.btnSetSources = QPushButton("Set Source IDs")
        self.btnSetSources.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btnSetSources.setStyleSheet(button_stylesheet)
        self.btnSetSources.setMinimumWidth(500)
        layoutSetSourcesBtn.addWidget(self.btnSetSources)

        # Output Folder
        layoutOutputFolder = QHBoxLayout()
        layoutOutputFolder.setAlignment(Qt.AlignLeft)
        self.lblOutputFolder = QLabel("Output Folder: ")
        self.lblOutputFolder.setFixedWidth(130)
        self.lblOutputFolder.setMinimumWidth(130)
        self.editOutputFolder = QLineEdit("")
        self.editOutputFolder.setReadOnly(True)
        self.editOutputFolder.setPlaceholderText(self.temp_folder)
        self.editOutputFolder.setStyleSheet("background-color: rgb(40,40,40); border: 1px solid rgb(90,90,90)")
        self.btnOutputFolder = QPushButton("...")
        self.btnOutputFolder.setFixedWidth(20)
        self.btnOutputFolder.setMinimumWidth(20)

        layoutOutputFolder.addWidget(self.lblOutputFolder)
        layoutOutputFolder.addWidget(self.editOutputFolder)
        layoutOutputFolder.addWidget(self.btnOutputFolder)

        # Tile size (combobox, power of 2)
        # TODO Add tile size limit (CoralNet allows 8k x 8k)

        # Delete temporary data (check box)
        # TODO boolean to delete temporary data in self.output_dir (make sure it's the right folder)

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

        # -------------------------
        # Parameters Layout
        # -------------------------
        layoutParameters = QVBoxLayout()

        layoutParameters.addLayout(layoutUsername)
        layoutParameters.addLayout(layoutPassword)
        layoutParameters.addLayout(layoutAuthorizationBtn)
        layoutParameters.addLayout(layoutSourceID1)
        layoutParameters.addLayout(layoutSourceID2)
        layoutParameters.addLayout(layoutSetSourcesBtn)
        layoutParameters.addLayout(layoutOutputFolder)
        layoutParameters.addWidget(self.console_widget)
        layoutParameters.addLayout(layoutButtons)

        layoutMiddlePanel = QVBoxLayout()
        layoutMiddlePanel.addLayout(layoutParameters)

        # ----------------------------
        # Label Mapping (Right Panel)
        # ----------------------------

        # Header Label
        header_label = QLabel("Label Mapping")
        header_label.setStyleSheet("font-weight: bold; font-size: 24px; color: white;")
        header_label.setAlignment(Qt.AlignCenter)

        # Create Import Mapping Button
        import_mapping_button = QPushButton("Import Mapping")
        import_mapping_button.clicked.connect(self.taglabImportMapping)

        # Create Save Mapping Button
        export_mapping_button = QPushButton("Export Mapping")
        export_mapping_button.clicked.connect(self.taglabExportMapping)

        # Create Horizontal Layout for Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(import_mapping_button)
        buttons_layout.addWidget(export_mapping_button)
        buttons_layout.addStretch(1)
        buttons_layout.setAlignment(Qt.AlignRight)

        # Scroll Area for Source List Widget
        self.label_mapping_scroll_area = QScrollArea()
        self.label_mapping_scroll_area.setWidgetResizable(True)
        self.label_mapping_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.label_mapping_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Widget to contain all the mapping rows
        self.mapping_widget = QWidget()
        self.mapping_layout = QVBoxLayout(self.mapping_widget)
        self.label_mapping_scroll_area.setWidget(self.mapping_widget)

        # Source List Widget Layout
        self.layoutRightPanel = QVBoxLayout()

        # -------------------------
        # Right Panel Layout
        # -------------------------

        # Add header, buttons, and mapping widget to the layout
        self.layoutRightPanel.addWidget(header_label)
        self.layoutRightPanel.addWidget(self.label_mapping_scroll_area)
        self.layoutRightPanel.addLayout(buttons_layout)

        # ---------------------------
        # Final Layout
        # ---------------------------

        layoutFinal = QHBoxLayout()
        # Source List Panel
        layoutFinal.addLayout(self.layoutLeftPanel)
        # Login, Console Panel
        layoutFinal.addLayout(layoutMiddlePanel)
        # Mapping Label Panel
        layoutFinal.addLayout(self.layoutRightPanel)

        self.setLayout(layoutFinal)

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

    def taglabLoadSourcesPanel(self):
        """
        Load sources panel with data from CoralNet sources.
        """
        try:
            # Get the sources from CoralNet
            self.driver, self.source_list = download_coralnet_sources(self.driver, None)

            # Clear existing items in the source list table
            self.source_list_table.setRowCount(0)

            # Add items dynamically to the table
            for i, r in self.source_list.iterrows():
                self.source_list_table.insertRow(i)
                self.source_list_table.setItem(i, 0, QTableWidgetItem(r['Source_ID']))
                self.source_list_table.setItem(i, 1, QTableWidgetItem(r['Source_Name']))
                self.source_list_table.setItem(i, 2, QTableWidgetItem(r['Source_URL']))

                # Set items in the row to be non-editable
                for col in range(self.source_list_table.columnCount()):
                    item = self.source_list_table.item(i, col)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        except Exception as e:
            raise Exception(f"Failed load Source list. {e}")

    @pyqtSlot()
    def coralnetAuthenticate(self):
        """
        Authenticates the username and password before starting the process.
        """
        # Good or Bad box
        msgBox = QMessageBox()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.username = self.editUsername.text()
        self.password = self.editPassword.text()

        try:

            if "" in [self.username, self.password]:
                raise Exception("You must provide valid CoralNet credentials.")

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

            # Autofill Source ID panel
            self.taglabLoadSourcesPanel()

            # Change the other fields from ReadOnly
            self.editSourceID1.setReadOnly(False)
            self.editSourceID2.setReadOnly(False)
            self.editOutputFolder.setReadOnly(False)
            self.editOutputFolder.setText(self.temp_folder)
            self.btnOutputFolder.clicked.connect(self.chooseOutputFolder)
            self.btnSetSources.clicked.connect(self.taglabLoadLabelsetPanel)

        except Exception as e:
            msgBox.setText(f"Could not complete authentication. {e}")
            msgBox.exec()

        self.console_widget.clearConsole()
        QApplication.restoreOverrideCursor()

    def addRowToLabelMapping(self, left_label, right_label):
        """

        """
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)

        left_text_box = QLineEdit(left_label)
        left_text_box.setReadOnly(True)
        left_text_box.setMinimumWidth(150)
        left_text_box.setMaximumWidth(200)
        row_layout.addWidget(left_text_box)

        arrow_label = QLabel("--->")
        arrow_label.setAlignment(Qt.AlignCenter)
        row_layout.addWidget(arrow_label)

        right_combo_box = QComboBox()
        right_combo_box.setEditable(True)
        right_combo_box.setMinimumWidth(150)
        right_combo_box.setMaximumWidth(200)
        right_combo_box.addItems([right_label] + self.taglab_labels)
        row_layout.addWidget(right_combo_box)

        # Add the row
        self.mapping_layout.addWidget(row_widget)

    @pyqtSlot()
    def taglabLoadLabelsetPanel(self):
        """

        """
        # Good or Bad box
        msgBox = QMessageBox()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.source_id_1 = int(self.editSourceID1.text())
        self.source_id_2 = int(self.editSourceID2.text())

        try:
            if "" in [self.source_id_1, self.source_id_1]:
                raise Exception("You must provide valid CoralNet Source IDs.")

            # Download the labelset for the source user provided
            self.driver, self.labelset = download_labelset(self.driver, self.source_id_2, None)

            if self.labelset is None:
                raise Exception(f"Source {self.source_id_2} does not have a labelset")

            # Clear the mapping layout
            while self.mapping_layout.count():
                item = self.mapping_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # Add each row to the widget
            for i, r in self.labelset.iterrows():
                self.addRowToLabelMapping(r['Name'], r['Name'])

        except Exception as e:
            msgBox.setText(f"Could not load labelset. {e}")
            msgBox.exec()

        self.console_widget.clearConsole()
        QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def taglabImportMapping(self):
        """

        """
        filters = "JSON (*.json)"
        header = f"Open {self.source_id_2} Mapping"
        filename, _ = QFileDialog.getOpenFileName(self, header, self.output_folder, filters)

        if filename:
            # Make sure it exists
            if not os.path.exists(filename):
                raise Exception(f"{filename} does not exist")

            # Open the existing JSON file
            with open(filename, 'r') as file:
                mapping_data = json.load(file)

            # Clear the mapping layout before importing
            while self.mapping_layout.count():
                item = self.mapping_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # Loop through and autofill
            for mapping in mapping_data:
                # Get the corresponding mapping
                left_label = mapping.get("CoralNet_Label", "")
                right_label = mapping.get("TagLab_Label", "")
                # Add row to mapping table
                self.addRowToLabelMapping(left_label, right_label)

    @pyqtSlot()
    def taglabExportMapping(self):
        """

        """
        filters = "JSON (*.json)"
        header = f"Save {self.source_id_2} Mapping"
        filename, _ = QFileDialog.getSaveFileName(self, header, self.output_folder, filters)

        if filename:
            mapping_data = []
            # Loop through the JSON file
            for index in range(self.mapping_layout.count()):
                # Get the current row
                row_widget = self.mapping_layout.itemAt(index).widget()
                if row_widget:
                    # Extract the CoralNet and TagLab labels
                    left_combo_box = row_widget.findChild(QLineEdit)
                    right_combo_box = row_widget.findChild(QComboBox)
                    # Add to json
                    if left_combo_box and right_combo_box:
                        mapping_data.append({
                            "CoralNet_Label": left_combo_box.text(),
                            "TagLab_Label": right_combo_box.currentText()
                        })
            # Write the JSON file
            with open(filename, 'w') as file:
                json.dump(mapping_data, file, indent=4)

            if os.path.exists(filename):
                msgBox = QMessageBox()
                msgBox.setText(f"File saved successfully")
                msgBox.exec()

    @pyqtSlot()
    def apply(self):
        """

        """
        # Good or Bad box
        msgBox = QMessageBox()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.output_folder = self.editOutputFolder.text()
        self.output_folder = f"{self.output_folder}/{get_now()}"

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