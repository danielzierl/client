import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
import cv2
import mediapipe as mp
import numpy as np
import pyqtgraph as pg
import json
import os
import csv


class FaceMeshWidget(QWidget):
    def __init__(self, saveToJson=False):
        super().__init__()
        self.saveToJson = saveToJson
        self.label =0
        # Setup layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setGeometry(200, 200, 600, 400)
        # Setup video display widget
        self.video_widget = pg.ImageView()
        self.video_widget.ui.histogram.hide()
        self.video_widget.ui.roiBtn.hide()
        self.video_widget.ui.menuBtn.hide()

        layout.addWidget(self.video_widget)


        # Start video capture
        self.capture = cv2.VideoCapture(0)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        # Setup face mesh detector
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def update_frame(self):
        # Get video frame
        ret, frame = self.capture.read()
        if not ret:
            return

        # Convert to RGB for face mesh detector
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run face mesh detector
        results = self.face_mesh.process(frame)

        # Get face landmarks
        self.xarr=[]
        self.yarr=[]
        self.zarr=[]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    self.x, self.y, self.z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), int(landmark.z)
                    self.xarr.append(self.x)
                    self.yarr.append(self.y)
                    self.zarr.append(self.z)
                    cv2.circle(frame, (self.x, self.y), 1, (0, 255, 0), -1)

        # Display video frame
        self.video_widget.setImage(np.rot90(frame, 1))
        if self.saveToJson:
            self.programBeSavin("data/data")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

    def programBeSavin(self,arg):
        self.xarr.extend(self.yarr)
        self.xarr.extend(self.zarr)
        mean = 0
        for el in self.xarr:
            mean+=el
        mean = mean/len(self.xarr)
        std=0
        for el in self.xarr:
            std += (el-mean)**2
        std = (std/len(self.xarr))**0.5
        for i,el in enumerate(self.xarr):
            self.xarr[i]= (el-mean)/std
        self.xarr.append(self.label)
        with open(f"{arg}.csv", 'a') as f:
            if self.xarr:
                writer = csv.writer(f)
                writer.writerow(self.xarr)
            
    


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup layout
        self.setWindowTitle("Facemesh Calibration")
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()

        # Setup buttons
        calibration_button = QPushButton("Calibration")
        calibration_button.clicked.connect(self.open_calibration_window)
        layout.addWidget(calibration_button)

        meas_button = QPushButton("Measure")
        meas_button.clicked.connect(self.open_meas_window)
        layout.addWidget(meas_button)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

        # Setup central widget
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_calibration_window(self):
        self.calibration_window = CalibrationWindow()
        self.calibration_window.show()

    def open_meas_window(self):
        self.calibration_window = ContinuousMeasuring()
        self.calibration_window.show()

    def run(self):
        print("Running...")

class CalibrationWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Setup layout
        self.setWindowTitle("Facemesh")
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()
        self.facemesh_widget = FaceMeshWidget()
        # Setup buttons
        normal_button = QPushButton("Normal")
        normal_button.clicked.connect(lambda: self.save_landmarks("normal",self.facemesh_widget))
        layout.addWidget(normal_button)

        face_button = QPushButton("Face")
        face_button.clicked.connect(lambda: self.save_landmarks("face",self.facemesh_widget))
        layout.addWidget(face_button)

        # Setup facemesh widget
        
        layout.addWidget(self.facemesh_widget)
        self.setLayout(layout)

    def save_landmarks(self, expression, facemesh):
        # Save landmarks to JSON file
        data = {
            "expression": expression,
            "landmarks": facemesh.landmarks
        }
        with open(f"{expression}.json", "w") as f:
            json.dump(data, f)

class ContinuousMeasuring(QWidget):
    def __init__(self):

        super().__init__()
        self.measuring = False
        self.emotion = False
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()
        self.facemesh_widget = FaceMeshWidget()

        self.normal_button = QPushButton("start")
        self.normal_button.clicked.connect(self.switchMode)

        self.label_button = QPushButton("no-emotion")
        self.label_button.clicked.connect(self.switchLabel)

        layout.addWidget(self.normal_button)
        layout.addWidget(self.label_button)

        

        layout.addWidget(self.facemesh_widget)
        
        self.setLayout(layout)
    
    
    def switchMode(self):
        self.measuring = not self.measuring
        self.normal_button.setText("stop" if self.measuring else "start")
        self.facemesh_widget.saveToJson = self.measuring

    def switchLabel(self):
        self.emotion = not self.emotion
        self.label_button.setText("emotion" if self.emotion else "no-emotion")
        self.facemesh_widget.label = int(self.emotion)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())