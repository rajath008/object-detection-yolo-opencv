
import cv2
import argparse
import numpy as np

items_found = []

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    items_found.append(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def perform_object_detection(frame):
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    cv2.imshow("Real-time Object Detection", frame)



def detect_objects(image_path):
    frame = cv2.imread(image_path)
    perform_object_detection(frame)
  
    output_path = "annotated.jpg"

    cv2.imwrite(output_path, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def perform_real_time_object_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        perform_object_detection(frame)

        if cv2.waitKey(1) == 13:  # 13 is the ASCII code for the Enter key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True,
                    help='path to YOLO config file')
    ap.add_argument('-w', '--weights', required=True,
                    help='path to YOLO pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,
                    help='path to text file containing class names')
    ap.add_argument('-i', '--image',
                    help='path to input image')
    args = ap.parse_args()

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    if args.image:
        detect_objects(args.image)
    else:
        perform_real_time_object_detection()

    i=1
    for item in items_found:
        print( f"{i}.{item}\n")
        i=i+1

#for real-time-obj-detection #python yolo_opencv.py  --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
#for static-object-detection #python yolo_opencv.py  --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --image dog.jpg