import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
import cv2
import mediapipe as mp
import numpy as np
import pyqtgraph as pg
import json
from operator import add
import os
import csv

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class FaceMeshWidget(QWidget):
    def __init__(self, saveToJson=False, classify_on_the_fly=False, classificationLabel=None):
        super().__init__()
        self.saveToJson = saveToJson
        self.classify_on_the_fly= classify_on_the_fly
        self.classification_label = classificationLabel
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

        #tmp data mean
        self.array = []

        #mean data
        self.mean_data = []

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
        if len(self.xarr)>0:
            self.array.append([self.xarr, self.yarr, self.zarr])
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

        # if len(self.array) == 5:
        #     self.mean_data.append(self.meanData())
        #     self.array = []


        # Display video frame
            self.video_widget.setImage(np.rot90(frame, 1))
            if self.classify_on_the_fly:
                from nnm import torch_model,save_path
                torch_model.to(device)
                torch_model.load_state_dict(torch.load(save_path))
                self.programBeClassifiin(torch_model)
            if self.saveToJson:
                self.programBeSavin("data/data")

    # def meanData(self):
    #     x_sum, y_sum, z_sum = np.zeros(468), np.zeros(468),np.zeros(468)
    #     for vector in self.array:
    #         vector = np.array(vector)
    #         x_sum = x_sum + (vector[0])
    #         y_sum = y_sum + (vector[1])
    #         z_sum = z_sum + (vector[2])
    #     x_avg = x_sum / 5
    #     y_avg = y_sum / 5
    #     z_avg = z_sum / 5
    #     return [x_avg, y_avg, z_avg]
    def closeEvent(self, event):
        self.capture.release()
        event.accept()

    def programBeClassifiin(self,torch_model):
        
        pre = torch.reshape(torch.from_numpy(np.array(self.xarr,dtype=float)), (1,-1)).to(torch.float32).to(device)
        
        out = torch_model(pre)
        _,predicted =torch.max((out), dim=1)
        self.classification_label.setText("no emotion" if  predicted == 0 else "emotion")


    def programBeSavin(self,arg):
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
        calibration_button = QPushButton("Photo booth")
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
        self.run_window = RunClassifiationWindow()
        self.run_window.show()

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

class RunClassifiationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.measuring = False
        self.emotion = False
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()
        self.labelino = QLabel("no emotion")
        self.facemesh_widget = FaceMeshWidget(classificationLabel=self.labelino, classify_on_the_fly=True)


        
        layout.addWidget(self.labelino)
        layout.addWidget(self.facemesh_widget)
        
        self.setLayout(layout)
    


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
