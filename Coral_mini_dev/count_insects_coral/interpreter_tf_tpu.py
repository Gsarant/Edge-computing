from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common

import pathlib
import os
import numpy as np
import count_insects_coral.init as init
from datetime import datetime

interpreter_tpu=None

def init_interpreter_tf(model_tflite__tpu_file_path):
  global interpreter_tpu
  starttime=datetime.now()
  interpreter_tpu = edgetpu.make_interpreter(model_tflite__tpu_file_path)
  interpreter_tpu.allocate_tensors()
  init.my_logs.info(f'Load Coral interpreter model {model_tflite__tpu_file_path}')
  endtime=datetime.now()
  init.my_logs.info(f'Initialization tpu time {(endtime-starttime).microseconds} microseconds ')
  



def interpreter_evaluation_tf(image):
  """ Image 240x240"""
  global interpreter_tpu
  starttime=datetime.now()
  scale, zero_point=common.input_details(interpreter_tpu, 'quantization')
  
  input_data= (np.float32(image) / 255.0) / (scale*1.0) + (zero_point*1.0)

  input_data = np.expand_dims(input_data, axis=2)

  common.set_input(interpreter_tpu, input_data)
  interpreter_tpu.invoke()
  
  output_data= common.output_tensor(interpreter_tpu, 0)
  
  
  
  scale, zero_point = interpreter_tpu.get_output_details()[0]['quantization']
  
  output_data = ((scale*1.0) * (output_data - zero_point*1.0))

  predictions=np.floor(np.array(output_data).item(0))

  predictions_round=int(np.around(np.array(output_data).item(0)))
  endtime=datetime.now()
  init.my_logs.info(f'Interpreter_tpu time {(endtime-starttime).microseconds} microseconds')
  print(f'output_data {output_data}')
  print(f'predictions {predictions}')
  print(f'predictions_round {predictions_round}')
  return predictions_round
