import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtNetwork
import cv2
import mediapipe as mp
import numpy as np
import pyqtgraph as pg
import json
from operator import add
import os
import csv
import time
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
        self.i=0
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
            if self.array is None:
                self.array = np.array([])
            if len(self.array) == 5:
                self.mean_data.append(self.meanData())
                self.array = np.array([])
            self.xarr.extend(self.yarr)
            self.xarr.extend(self.zarr)
            temp = np.array(self.xarr)
            self.array = np.concatenate((self.array, temp))
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



        # Display video frame
            self.video_widget.setImage(np.rot90(frame, 1))
            if self.classify_on_the_fly:
                from nnm import torch,save_path
                torch.to(device)
                torch.load_state_dict(torch.load(save_path))
                self.programBeClassifiin(torch)
                self.i +=1
            if self.saveToJson:
                self.programBeSavin("data/data")

    def meanData(self):
        self.array.view(-1,1404)
        return np.average(self.array, dim=0)
        
    def closeEvent(self, event):
        self.capture.release()
        event.accept()

    def programBeClassifiin(self,torch_model):
        pre =torch.tensor([])
        pre =torch.cat((pre, torch.reshape(torch.from_numpy(np.array(self.xarr,dtype=float)), (1,-1)).to(torch.float32).to(device)))
        if self.i>0:
            bias = 2
            self.i=0
            out = torch_model(pre)
            out = torch.mean(out, dim=0)
            for i in range(1,len(out)):
                out[i]-=bias

            _,predicted =torch.max((out), dim=0)
            print(out)
            
            self.classification_label.setText("emotion level:"+str(predicted.item()))
            # self.doRequest() if predicted ==1 else 0


    def programBeSavin(self,arg):
        self.xarr.append(self.label)
        with open(f"{arg}.csv", 'a') as f:
            if self.xarr:
                writer = csv.writer(f)
                writer.writerow(self.xarr)

    def doRequest(self):   
    
        url = "https://iot.benetronic.com/mymodule/tCNKNYAsyc/gvnF/promobilx1@gmail.com/132/0/0"
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        
        self.nam = QtNetwork.QNetworkAccessManager()
        
        self.nam.get(req)  
            


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
        self.emotion = 0
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout()
        self.facemesh_widget = FaceMeshWidget()
        self.label = QLabel("0")

        self.normal_button = QPushButton("start")
        self.normal_button.clicked.connect(self.switchMode)

        self.label_button = QPushButton("no-emotion")
        self.label_button.clicked.connect(lambda :self.setEm(0))

        self.label2_button = QPushButton("emotion")
        self.label2_button.clicked.connect(lambda :self.setEm(1))

        self.label3_button = QPushButton("epic-emotion")
        self.label3_button.clicked.connect(lambda :self.setEm(2))

        layout.addWidget(self.label)
        layout.addWidget(self.normal_button)
        layout.addWidget(self.label_button)
        layout.addWidget(self.label2_button)
        layout.addWidget(self.label3_button)

        

        layout.addWidget(self.facemesh_widget)
        
        self.setLayout(layout)
    
    def setEm(self,num):
        self.emotion=num
        self.label.setText(str(self.emotion))
        self.facemesh_widget.label = self.emotion

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
