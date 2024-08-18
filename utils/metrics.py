import gzip
import io
import time
from typing import Dict, Tuple
import numpy as np
import torch
from torchinfo import summary
import onnx
import onnxruntime as ort

def get_num_macs(model) -> int:
    res = summary(model, (1, 3, 320, 320), verbose=0)
    return res.total_mult_adds

def get_num_params(model) -> int:
    res = summary(model, (1, 3, 320, 320), verbose=0)
    return res.total_params

def get_time(model, device, iterations=100) -> float:
    model.eval()
    
    input = torch.rand(1, 3, 320, 320)
    input = input.to(device)

    times = []
    for _ in range(iterations):
        t_start = time.time()
        if device == 'cuda': torch.cuda.synchronize()
        out = model(input)
        if device == 'cuda': torch.cuda.synchronize()
        t_end = time.time()
        if device == 'cuda': torch.max(out)
        times.append(t_end - t_start)
        
    # times = list(sorted(times, reverse=False))
        
    return np.mean(times) * 1000 #ms

def get_time_onnx(device, load_path, num_classes, iterations=100):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(f"{load_path}.onnx", sess_options, providers=['CUDAExecutionProvider'])
    binding = session.io_binding()

    output_orig = torch.empty((1, 5, 10, 10, 5 + num_classes), dtype=torch.float32, device=torch.device("cpu"))
    
    binding.bind_output(
        name = 'output',
        device_type = "cpu",
        device_id = 0,
        element_type = np.float32,
        shape = output_orig.shape,
        buffer_ptr = output_orig.data_ptr()
    )
    
    input = torch.rand(1, 3, 320, 320)
    input = input.to(device)
    
    binding.bind_input(
            name = 'input',
            device_type = "cuda",
            device_id = 0,
            element_type = np.float32,
            shape = input.shape,
            buffer_ptr = input.data_ptr()
        )
    
    times = []
    for _ in range(iterations):
        t_start = time.time()
        session.run_with_iobinding(binding)
        t_end = time.time()
        times.append(t_end - t_start)
        
    return np.mean(times) * 1000
    

def get_size_on_disk(state_dict: Dict) -> Tuple[int, int]:
    buff = io.BytesIO()
    torch.save(state_dict, buff)
    # compressed_buff = gzip.compress(buff.getvalue(), compresslevel=9)
    # compressed_buff.__sizeof__(), 
    return buff.__sizeof__() / 1000000 # MB