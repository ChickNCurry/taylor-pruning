from flask import Flask, Response, send_file
from threading import Thread
from queue import Queue
import time
import cv2
import torch
from utils.utils import apply_bboxes, apply_fps
from utils.camera import CameraDisplay, CameraDetectionDisplay
from utils.yolo import non_max_suppression, nms, filter_boxes
from torchvision.transforms.functional import to_tensor
import onnx
import onnxruntime as ort
import numpy as np
import json

log_frames = 0
app = Flask('')
last_image = 0
cam = None
net = None

link1 = Queue(1)
link2 = Queue(1)
link3 = Queue(1)
link4 = Queue(1)

def pipeline0():
    while(True):
        task = link1.get()
        self = task['self']
        image = task['image']

        s = time.time()

        input = to_tensor(image).unsqueeze(0).to(self.device)
        output = torch.empty((1, 5, 10, 10, 5 + self.classes), dtype=torch.float32, device='cpu')

        self.binding.bind_input(
            name = 'input',
            device_type = self.device_name,
            device_id = 0,
            element_type = np.float32,
            shape = input.shape,
            buffer_ptr = input.data_ptr()
        )

        self.binding.bind_output(
            name = 'output',
            device_type = 'cpu',
            device_id = 0,
            element_type = np.float32,
            shape = output.shape,
            buffer_ptr = output.data_ptr()
        )

        self.session.run_with_iobinding(self.binding)

        e = time.time()

        link2.put({'image': image, 'output': output, 'dt_model': e - s})

def pipeline1():
    while(True):
        task = link2.get()

        #filter boxes based on confidence score (class_score*confidence) and overlap
        s = time.time()
        task['bboxes'] = non_max_suppression(task['output'], conf_thres=0.25, iou_thres=0.30)
        #task['bboxes'] = nms(filter_boxes(task['output'], 0.25), 0.30)
        e = time.time()
        
        del task['output']
        task['dt_nms'] = e - s

        link3.put(task)

def pipeline2():
    global last_image, log_frames
    frame_times = []
    json_log = []

    while(True):
        task = link3.get()

        s = time.time()
        dt = s - last_image
        last_image = s

        frame_times.append(dt)

        if(len(frame_times) > 30):
            frame_times.pop(0)

        avg_dt = sum(frame_times) / len(frame_times)

        box_count = apply_bboxes(task['image'], task['bboxes'])
        apply_fps(task['image'], avg_dt)

        ret, buf = cv2.imencode('.jpg', task['image'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        res = buf.tobytes()
        e = time.time()

        if log_frames > 0:
            log_frames -= 1
            json_log.append({'dt_throughput': dt, 'dt_model': task['dt_model'], 'dt_nms': task['dt_nms'], 'dt_encode': e - s, 'bboxes': box_count})

            if log_frames == 0:
                with open('frame_times.json', 'w') as f:
                    json.dump(json_log, f, ensure_ascii=False)
                print('wrote results to file')

        link4.put(res)

class InferenceModel:
    def __init__(self, device_name, model_path, classes):
        self.device = torch.device(device_name)
        self.device_name = device_name
        self.classes = classes

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, options, providers=['CUDAExecutionProvider'])
        self.binding = self.session.io_binding()

    def get_next(self):
        global link4
        return link4.get()
    
    def run(self, image):
        global link1
        
        if not link1.full():
            link1.put({'self': self, 'image': image})

def gen_frames():
    global net
    while True:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + net.get_next() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam_start')
def cam_start():
    global cam, net, callback, last_image
    res = 'Camera started!'

    try:
        if cam is None:
            cam = CameraDetectionDisplay(net, callback, True, False)
            cam.start()
            last_image = time.time()
    except Exception as e:
        cam = None
        res = repr(e)

    return res

@app.route('/cam_stop')
def cam_stop():
    global cam
    res = 'Camera stopped!'

    try:
        if cam is not None:
            cam.stop()
            cam.release()
    except Exception as e:
        res = repr(e)
    finally:
        cam = None

    return res

@app.route('/log_start')
def log_start():
    global log_frames
    log_frames = 1000
    return f'Logging started for {log_frames} frames'

@app.route('/')
def index():
    return send_file('web/index.html')

@app.route('/bootstrap.min.css')
def bootstrap():
    return send_file('web/bootstrap.min.css')

def callback(_, image):
    global net

    crop_size = 320
    w, h = image.shape[0], image.shape[1]
    sx = int((w - crop_size) / 2)
    sy = int((h - crop_size) / 2)
    net.run(image[sx:(sx + crop_size), sy:(sy + crop_size), :])

def start_server(device_name, model_path, classes):
    p0 = Thread(target=pipeline0)
    p1 = Thread(target=pipeline1)
    p2 = Thread(target=pipeline2)
    p0.start()
    p1.start()
    p2.start()

    print('pipeline setup!')

    global net
    net = InferenceModel(device_name, model_path, classes)
    print('model loaded')

    net.run(np.empty((320, 320, 3), dtype=np.uint8))
    print('pipeline filled!')

    app.run(host='0.0.0.0')
    print('server started')

start_server('cuda', 'runs/taylor2/voc_pruned_6_finetuned.pt.onnx', 1)
