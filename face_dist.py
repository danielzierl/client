from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
# import model
import torch.nn as nn
import copy
import os
from utils import merge
from model import CustomMobileNet, CustomNet
from nnm import load_data

save_path = "saved_model"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
file = "./data/data.csv"
file_dir = "./data/new/dataset_1"

NUM_CLASSES = 4
NUM_FEATS = 1404
num_lm = 224
data_shape = (468, 3)

def L2_distance_matrix(A, B):
    # (X - Y)^2 = X^2 -2XY + Y^2
    dists = np.sum(A ** 2, axis=1).reshape(-1, 1) - 2*(A@B.T) + np.sum(B**2, axis=1)
    return dists

def make_averaged_matrix():
    data = merge(file_dir)

    y = data[:, -1].astype(int)
    data = data[:, :-1].reshape(len(data), 468, 3)

    num_cls = max(y) + 1

    distMatricesX = [[] for _ in range(num_cls)]
    distMatricesY = [[] for _ in range(num_cls)]
    distMatricesZ = [[] for _ in range(num_cls)]

    for face_data, cls_value in zip(data, y):
        face_data = face_data[:, np.newaxis]
        distMatricesX[cls_value].append(L2_distance_matrix(face_data[0], face_data[0]))
        distMatricesX[cls_value].append(L2_distance_matrix(face_data[1], face_data[1]))
        distMatricesX[cls_value].append(L2_distance_matrix(face_data[2], face_data[2]))

    averagedMatricesX = []
    averagedMatricesY = []
    averagedMatricesZ = []
    dims = [distMatricesX, distMatricesY, distMatricesZ]
    averaged_dims = [averagedMatricesX, averagedMatricesY, averagedMatricesZ]
    for dim, avg_dim in zip(dims, averaged_dims):
        averagedMatrix = np.average(dim, axis=0)
        averaged_dims.append(averagedMatrix)

    return averaged_dims

import cv2
import mediapipe as mp
import time

OUTPUT_PATH = "../data/test.json"


def compare(finger_prints, current_data):
    face_data = current_data[:, np.newaxis]
    current_dist_matrix = L2_distance_matrix(face_data, face_data)

    diffs = []
    for finger_print in finger_prints:
        diff_loss = np.diff(finger_print, current_dist_matrix)
        diffs.append(diff_loss)

    min_cls = np.argmin(diffs)
    return min_cls


def video():
    averaged_matrices = make_averaged_matrix()

    cap = cv2.VideoCapture(0)
    pTime = 0
    NUM_FACE = 1

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

                current_face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)

                    current_face.append([x, y, lm.z])

                current_face = np.array(current_face).T
                min_dist_cls = compare(averaged_matrices, current_face)
                print(min_dist_cls)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Test", img)

        if cv2.waitKey(1)& 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    video()




