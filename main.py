from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5 import QtTest
import mediapipe as mp
import numpy as np
import os
import pickle
import pyttsx3
import threading
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.isShowingCam = True
        self.isDetectingHand = False
        self.isPredicting = False
        uic.loadUi('mainUI.ui', self)
        self.setWindowTitle("Sign Language Detection")
        
        self.pushButton.clicked.connect(self.detectHand)
        self.pushButton_2.clicked.connect(self.showCam)
        self.pushButton_3.clicked.connect(self.predictHand)
        self.pushButton_4.clicked.connect(self.noPredict)
        self.menuAdd.triggered.connect(self.addData)
        
        ##PICKLE DATA
        self.mphands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDraingStyle = mp.solutions.drawing_styles
        self.hands = self.mphands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.model_dict = pickle.load(open("modelASL.pickle", "rb"))
        self.label_dict = {"I": "I", "L": "LOVE", "U": "U", "A": "A", "B": "B", "C": "C"}
        self.model = self.model_dict["model"]
        
        ##TEXT2SPEECH
        self.voiceEngine = pyttsx3.init('sapi5')
        self.voices = self.voiceEngine.getProperty('voices')
        self.voiceEngine.setProperty('voice', self.voices[0].id)
        
        ##VARIABLES
        self.predictedText = ""
        self.isSpeaking = False
        self.isAddData = False
        self.folderName = None
        self.savedFrame = 0
        
        self.show()
        cap = cv2.VideoCapture('sign2.mp4')
        self.frame_count = 0
        self.frame_rate = 30
        while True:
            ret, frame = cap.read()
            
            if self.isShowingCam:
                frameCount = 0
                self.label_3.setText("Showing Cam")
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.isPredicting:
                    data_aux = []
                    x_ = []
                    y_ = []
                    result = self.hands.process(img)
                    if result.multi_hand_landmarks:
                        for handLandmarks in result.multi_hand_landmarks:
                            for i in range(len(handLandmarks.landmark)):
                                x = handLandmarks.landmark[i].x
                                y = handLandmarks.landmark[i].y
                                data_aux.append(x)
                                data_aux.append(y)
                                x_.append(x)
                                y_.append(y)
                        data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
                        pred = self.model.predict([np.asarray(data_aux)])
                        self.predictedText = pred[0]
                        self.label_2.setText(self.predictedText)
                        
                        if not self.isSpeaking:
                            self.PlaySound(self.predictedText)
                if self.isAddData:
                        self.label_3.setText(f"Adding Data {self.folderName} : count {self.savedFrame}")
                        cv2.imwrite(f"data/{self.folderName}/{self.folderName}_{self.savedFrame}.jpg", frame)
                        if self.savedFrame == 300:
                            self.isAddData = False
                            self.savedFrame = 0
                        self.savedFrame += 1
                q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                vid = QPixmap(q_image).scaled(self.label.size(), Qt.KeepAspectRatio)
                self.label.setPixmap(vid)
            elif self.detectHand:
                self.label_3.setText("Detecting Hand")
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb_img)
                data_aux = []
                x_ = []
                y_ = []
                if result.multi_hand_landmarks:
                    for handLandmarks in result.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(rgb_img,
                                            handLandmarks,
                                            self.mphands.HAND_CONNECTIONS, 
                                            self.mpDraingStyle.get_default_hand_landmarks_style(), 
                                            self.mpDraingStyle.get_default_hand_connections_style())
                    if self.isPredicting:
                        for i in range(len(handLandmarks.landmark)):
                                x = handLandmarks.landmark[i].x
                                y = handLandmarks.landmark[i].y
                                data_aux.append(x)
                                data_aux.append(y)
                                x_.append(x)
                                y_.append(y)
                        
                        data_aux = np.pad(data_aux, (0, 84 - len(data_aux)), 'constant')
                        pred = self.model.predict([np.asarray(data_aux)])
                        self.predictedText = pred[0]
                        self.label_2.setText(self.predictedText)
                        
                        
                        if not self.isSpeaking:
                            self.PlaySound(self.predictedText)
                    if self.isAddData:
                        self.label_3.setText(f"Adding Data {self.folderName} : count {self.savedFrame}")
                        cv2.imwrite(f"data/{self.folderName}/{self.folderName}_{self.savedFrame}.jpg", frame)
                        self.savedFrame += 1
                        if self.savedFrame == 300:
                            self.isAddData = False
                            self.savedFrame = 0
                        
                height, width, channel = rgb_img.shape
                bytes_per_line = 3 * width
                q_image = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                vid = QPixmap(q_image).scaled(self.label.size(), Qt.KeepAspectRatio)
                self.label.setPixmap(vid)
            self.frame_count += 1
            QtTest.QTest.qWait(int(1000/60))
    def detectHand(self):
        self.isDetectingHand = True
        self.isShowingCam = False
        
    def showCam(self):
        self.isShowingCam = True
        self.isDetectingHand = False
        
        
    def predictHand(self):
        self.isPredicting = True
        
    def noPredict(self):
        self.isPredicting = False
    
    def PlaySound(self, text):
        thread = threading.Thread(target=self.speak, args=text)
        thread.start()
    def speak(self, text):
        self.isSpeaking = True
        self.voiceEngine.say(text)
        self.voiceEngine.runAndWait()
        self.isSpeaking = False
        
    def addData(self):
        self.isPredicting = False
        foldername = QInputDialog.getText(self, 'Input Dialog', 'Enter Folder Name', QLineEdit.Normal)
        if foldername[0]:
            self.folderName = foldername[0]
            os.makedirs(f'data/{foldername[0]}')
            self.isAddData = True
app = QApplication([])
window = MainWindow()
app.exec_()