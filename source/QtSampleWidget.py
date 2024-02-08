from PyQt5.QtCore import Qt, QSize, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QPainter, QImage, QPixmap, QIcon, qRgb, qRed, qGreen, qBlue
from PyQt5.QtWidgets import QComboBox, QSizePolicy, QLineEdit, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QSlider, QGroupBox, QMessageBox, QCheckBox, QWidget, QDialog, QFileDialog, QStackedWidget

from source.Annotation import Annotation
from source.QtWorkingAreaWidget import QtWorkingAreaWidget

import numpy as np


class QtSampleWidget(QWidget):
    # choosedSample = pyqtSignal(int)
    closewidget = pyqtSignal()
    validchoices = pyqtSignal()

    def __init__(self, parent=None):
        super(QtSampleWidget, self).__init__(parent)

        self.choosednumber = None
        self.offset = None

        self.setStyleSheet("background-color: rgb(40,40,40); color: white")

        layoutHM = QHBoxLayout()

        self.lblMethod = QLabel("Sampling Method: ")

        self.comboMethod = QComboBox()
        self.comboMethod.setMinimumWidth(300)
        self.comboMethod.addItem('Grid Sampling')
        self.comboMethod.addItem('Uniform Sampling')

        layoutHM.addWidget(self.lblMethod)
        layoutHM.addWidget(self.comboMethod)

        layoutHN = QHBoxLayout()
        self.lblNumber = QLabel("Number Of Points: ")
        self.editNumber = QLineEdit()
        self.editNumber.setPlaceholderText("Type Number Of Point")

        layoutHN.addWidget(self.lblNumber)
        layoutHN.addWidget(self.editNumber)

        layoutHOFF = QHBoxLayout()
        self.lblOFF = QLabel("Offset (px): ")
        self.editOFF = QLineEdit()
        self.editOFF.setPlaceholderText("Type pixels of offset")

        layoutHOFF.addWidget(self.lblOFF)
        layoutHOFF.addWidget(self.editOFF)

        layoutInfo = QVBoxLayout()
        layoutInfo.setAlignment(Qt.AlignLeft)
        layoutInfo.addLayout(layoutHM)
        layoutInfo.addLayout(layoutHN)
        layoutInfo.addLayout(layoutHOFF)

        # Create a stacked widget to hold Select Area Widget
        layoutHA = QHBoxLayout()

        self.lblWorkingArea = QLabel('Select Working Area')
        self.lblWorkingArea.setMinimumWidth(300)
        # Set the alignment to take up the entire width
        self.lblWorkingArea.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # Set the font to bold
        font = self.lblWorkingArea.font()
        font.setBold(True)
        self.lblWorkingArea.setFont(font)
        layoutHA.addWidget(self.lblWorkingArea)
        layoutHA.setAlignment(Qt.AlignCenter)
        layoutHA.addStretch()

        # Create a working area widget, connect only what's needed for point sampling
        self.working_area_widget = QtWorkingAreaWidget(self)
        self.working_area_widget.btnChooseArea.clicked.connect(self.parent().enableAreaSelection)
        self.working_area_widget.btnApply.clicked.connect(self.parent().setWorkingArea)
        selection_tool = self.parent().activeviewer.tools.tools["SELECTAREA"]
        selection_tool.setAreaStyle("WORKING")
        selection_tool.rectChanged[int, int, int, int].connect(self.working_area_widget.updateArea)
        self.working_area_widget.areaChanged[int, int, int, int].connect(selection_tool.setSelectionRectangle)

        # Initialize the working area to the entire screen
        wa = [0, 0, self.parent().activeviewer.image.width, self.parent().activeviewer.image.height]
        self.working_area_widget.updateArea(wa[1], wa[0], wa[2], wa[3])

        # These are needed, as the working area values are read from the Label boxes
        self.working_area_widget.btnCancel.setVisible(False)
        self.working_area_widget.btnApply.setVisible(False)
        self.working_area_widget.btnDelete.setVisible(False)

        # Stacked widget contains the working area widget
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.working_area_widget)

        layoutHB = QHBoxLayout()

        self.btnCancel = QPushButton("Cancel")
        self.btnCancel.clicked.connect(self.close)
        self.btnOK = QPushButton("Apply")
        self.btnOK.clicked.connect(self.apply)
        layoutHB.setAlignment(Qt.AlignRight)
        layoutHB.addStretch()
        layoutHB.addWidget(self.btnCancel)
        layoutHB.addWidget(self.btnOK)

        layout = QVBoxLayout()
        layout.addLayout(layoutInfo)
        layout.addSpacing(20)
        layout.addLayout(layoutHA)
        layout.addWidget(self.stacked_widget)
        layout.addLayout(layoutHB)
        self.setLayout(layout)

        self.setWindowTitle("Sampling Settings")
        self.setWindowFlags(Qt.Window |
                            Qt.CustomizeWindowHint |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowTitleHint)

    @pyqtSlot()
    def apply(self):

        if self.editNumber.text() == "" or self.editNumber.text() == 0 or self.editNumber.text().isnumeric() == False:
            msgBox = QMessageBox()
            msgBox.setText("Please, indicate the number of sampled points.")
            msgBox.exec()
            return
        else:
            self.choosednumber = int(self.editNumber.text())

        if self.editOFF.text() == "" or self.editOFF.text().isnumeric() == False:
            self.offset = 0
        else:
            self.offset = int(self.editOFF.text())

        self.validchoices.emit()

    def closeEvent(self, event):
        self.closewidget.emit()
        self.working_area_widget.closeEvent(event)
        super(QtSampleWidget, self).closeEvent(event)
