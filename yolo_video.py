import argparse, cv2, imutils, numpy as np, os, time
from datetime import datetime, timedelta
from tabulate import tabulate

np.random.seed(42)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

LABELS = open("model/coco.names").read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
total = int(vs.get(prop))

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes, confidences, classIDs = [], [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores[classID]

            if np.argmax(scores) == 2 and confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(np.argmax(scores))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, "Current # cars: {}".format(len(idxs)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            completion_time = datetime.now() + timedelta(seconds=elap * total)
            print(tabulate([
                ["Total video frames", "{}".format(total)],
                ["Detection time per frame", "{:.4f} seconds".format(elap)],
                ["Estimated duration", "{:.4f} seconds".format(elap * total)],
                ["Estimated completion time", completion_time.strftime("%H:%M:%S")]
            ]))

    writer.write(frame)

print("\nDetection complete. Program terminated.")
writer.release()
vs.release()
