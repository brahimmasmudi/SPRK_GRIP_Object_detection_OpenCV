import cv2
config='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen='frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen,config)
classlabels = []
file = 'Oldlabels.txt'
with open(file,'rt') as f:
    classlabels = f.read().rstrip('\n').split('\n')
print(classlabels)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
cap = cv2.VideoCapture('untitled.webm')



if not cap.isOpened():
    raise IOError("Cannot open Video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    size=(900,700)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5,nmsThreshold=0.55)
    print(confidence)

    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd <= 80):
                cv2.rectangle(frame, boxes, (0, 255, 0), 1)
                cv2.putText(frame, classlabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 0, 255), thickness=2)

    cv2.putText(frame,'Brahim Masmoudi : Spark foundation intern_JULY2022 Batch',(20, 20),font, 1,(255, 0, 255),2,cv2.LINE_4)
    cv2.imshow('_', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()