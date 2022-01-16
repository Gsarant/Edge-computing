from tflite_runtime.interpreter import Interpreter 
import pathlib
import os
import numpy as np
import count_insects_coral.init as init
from datetime import datetime

interpreter=None
height=None
width=None
input_details=None
output_details=None

def init_interpreter_tf(model_tflite_file_path):
  global input_details
  global output_details
  global interpreter
  starttime=datetime.now()
  interpreter = Interpreter(
      model_path=model_tflite_file_path, num_threads=None)

  init.my_logs.info(f'Load interpreter model {model_tflite_file_path}')
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  endtime=datetime.now()
  init.my_logs.info(f'Initialization time {(endtime-starttime).microseconds} microseconds ')
  



def interpreter_evaluation_tf(image):
  """ Image 240x240"""
  global input_details
  global output_details
  global interpreter
  starttime=datetime.now()

  scale, zero_point = input_details[0]['quantization']

  input_data= (np.float32(image) / 255.0) / (scale*1.0) + (zero_point*1.0)

  #input_data =np.array((np.float32(image)/255)).astype(input_details[0]["dtype"])
  input_data = np.expand_dims(input_data, axis=0).astype(input_details[0]["dtype"])
  input_data = np.expand_dims(input_data, axis=3).astype(input_details[0]["dtype"])
  

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output_data1 = interpreter.get_tensor(output_details[0]['index'])
  
  scale, zero_point = output_details[0]['quantization']
  
  output_data = ((scale*1.0) * (output_data1 - zero_point*1.0))

  predictions=np.floor(np.array(output_data).item(0))

  predictions_round=int(np.around(np.array(output_data).item(0)))
  endtime=datetime.now()
  init.my_logs.info(f'Interpreter time {(endtime-starttime).microseconds} microseconds')
  print(f'output_data1 {output_data1}')
  print(f'output_data {output_data}')
  print(f'predictions {predictions}')
  print(f'predictions_round {predictions_round}')
  return predictions_round
