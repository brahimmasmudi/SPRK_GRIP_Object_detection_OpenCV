
import cv2


config='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen='frozen_inference_graph.pb'
file='Oldlabels.txt'
labels=[]
with open(file,'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
print(labels)



model=cv2.dnn_DetectionModel(frozen,config)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
image=cv2.imread('crosswalk.jpg')
ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.5,nmsThreshold=0.55)
print(confidence)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd <= 80):
                cv2.rectangle(image, boxes, (0, 255, 0), 1)
                cv2.putText(image,labels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 0, 255), thickness=2)

    cv2.putText(image,'Brahim Masmoudi : Spark foundation intern_JULY2022 Batch',(20, 20),font, 1,(255, 0, 255),2,cv2.LINE_4)
    cv2.imshow('_',image)
    c=cv2.waitKey(1)
    if c==27:
        break
cv2.destroyAllWindows()


