import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtNetwork
import cv2
import collections

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service

import mediapipe as mp
import numpy as np
from selenium.webdriver.common.keys import Keys
import pyqtgraph as pg
import json
from operator import add
import os
import pdfreader
import csv
import time
import torch
import pyautogui
import mouse

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class FaceMeshWidget(QWidget):
    def __init__(self, saveToJson=False, classify_on_the_fly=False, classificationLabel=None):
        super().__init__()
        self.saveToJson = saveToJson
        self.classify_on_the_fly = classify_on_the_fly
        self.classification_label = classificationLabel
        self.label = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.x = 1000
        self.y = 1000
        self.i = 0
        # Setup layout
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.setGeometry(200, 200, 600, 400)
        # Setup video display widget
        self.video_widget = pg.ImageView()
        self.video_widget.ui.histogram.hide()
        self.video_widget.ui.roiBtn.hide()
        self.video_widget.ui.menuBtn.hide()

        self.epic_queue = collections.deque(maxlen=6)

        self.pdfReader = pdfreader.pdfReader
        chromedriver_path = "./chromedriver.exe"
        service = Service(chromedriver_path)

        # Vytvoření instance třídy webdriver.Chrome s použitím objektu Service
        self.driver = webdriver.Chrome(service=service)
        # self.driver = webdriver.Chrome('C:/Users/jakub/Desktop/chromedriver.exe')
        url = "file:///C:/Programming/Python Projects/client/book.pdf"
        # Otevření PDF souboru v prohlížeči
        self.driver.get(url)

        self.actions = ActionChains(self.driver)

        # Otevření PDF souboru v prohlížeči

        layout.addWidget(self.video_widget)

        # Start video capture
        self.capture = cv2.VideoCapture(0)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        # tmp data mean
        self.array = []

        # mean data
        self.mean_data = []

        # Setup face mesh detector
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()


        from classificators import combine_models
        self.model = combine_models()

        # Time
        self.delay = 2
        self.pdf_lock_time = 0
        self.lock_flag = False
        self.quited = False

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
        self.xarr = []
        self.yarr = []
        self.zarr = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for id, landmark in enumerate(face_landmarks.landmark):
                    # from nn import mouth_inds
                    self.x, self.y, self.z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), int(
                        landmark.z)
                    self.xarr.append(self.x)
                    self.yarr.append(self.y)
                    self.zarr.append(self.z)
                    cv2.circle(frame, (self.x, self.y), 1, (0, 255, 0), -1)
        if len(self.xarr) > 0:
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
                mean += el
            mean = mean / len(self.xarr)
            std = 0
            for el in self.xarr:
                std += (el - mean) ** 2
            std = (std / len(self.xarr)) ** 0.5
            for i, el in enumerate(self.xarr):
                self.xarr[i] = (el - mean) / std

            # Display video frame
            self.video_widget.setImage(np.rot90(frame, 1))
            if self.classify_on_the_fly:

                self.programBeClassifiin(self.model)

            if self.saveToJson:
                self.programBeSavin("data/data")

    def meanData(self):
        self.array.view(-1, 1404)
        return np.average(self.array, dim=0)

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

    def programBeClassifiin(self,torch_model):
        # from classificators import make_xgboost
        #
        # model = make_xgboost()
        preds = []
        for model in torch_model:
            preds.append(model.predict(np.array(self.xarr).reshape(1, -1)))

        preds = np.concatenate(preds).astype(int)

        for pred in preds:
            self.epic_queue.append(pred)

        print(preds)
        out_pred = max(set(self.epic_queue), key=self.epic_queue.count)


        # out_pred = np.bincount(preds).argmax()

        # pre = torch.reshape(torch.from_numpy(np.array(self.xarr,dtype=float)), (1,-1)).to(torch.float32).to(device)
        #
        # out = torch_model(pre)
        # _,predicted =torch.max((out), dim=1)
        self.classification_label.setText("no emotion" if out_pred == 0 else f"emotion {out_pred}")
            
        predicted = out_pred

        if self.lock_flag:
            current_t = time.time()
            if self.pdf_lock_time + self.delay < current_t:
                self.lock_flag = False
        else:
            self.pdf_lock_time = time.time()
            self.lock_flag = True

            if not self.quited:
                if predicted == 1:
                    self.pdfReader.on_key_release(self.actions, 1)
                if predicted == 2:
                    self.pdfReader.on_key_release(self.actions, 2)
                if predicted == 4:
                    self.pdfReader.on_key_release(self.actions, 3)
                if predicted == 3:
                    self.quited = True
                    self.driver.quit()
                    pass

        # self.driver.quit()

    def programBeSavin(self, arg):
        self.xarr.append(self.label)
        with open(f"{arg}.csv", 'a') as f:
            if self.xarr:
                writer = csv.writer(f)
                writer.writerow(self.xarr)

    def doRequest(self, url):

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
        normal_button.clicked.connect(lambda: self.save_landmarks("normal", self.facemesh_widget))
        layout.addWidget(normal_button)

        face_button = QPushButton("Face")
        face_button.clicked.connect(lambda: self.save_landmarks("face", self.facemesh_widget))
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
        self.label_button.clicked.connect(lambda: self.setEm(0))

        self.label2_button = QPushButton("emotion")
        self.label2_button.clicked.connect(lambda: self.setEm(1))

        self.label3_button = QPushButton("epic-emotion")
        self.label3_button.clicked.connect(lambda: self.setEm(2))

        self.label4_button = QPushButton("super-emotion")
        self.label4_button.clicked.connect(lambda: self.setEm(3))

        self.label5_button = QPushButton("super-emotion")
        self.label5_button.clicked.connect(lambda: self.setEm(4))

        layout.addWidget(self.label)
        layout.addWidget(self.normal_button)
        layout.addWidget(self.label_button)
        layout.addWidget(self.label2_button)
        layout.addWidget(self.label3_button)
        layout.addWidget(self.label4_button)
        layout.addWidget(self.label5_button)

        layout.addWidget(self.facemesh_widget)

        self.setLayout(layout)

    def setEm(self, num):
        self.emotion = num
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
