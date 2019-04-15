import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="/home/cpop/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model
interpreter.set_tensor(input_details[0]['index'], [np.float32(5)])
interpreter.set_tensor(input_details[1]['index'], [np.float32(0)])
interpreter.set_tensor(input_details[2]['index'], [np.float32(11)])
interpreter.set_tensor(input_details[3]['index'], [np.float32(0)])
interpreter.set_tensor(input_details[4]['index'], [np.float32(1)])
interpreter.set_tensor(input_details[5]['index'], [np.float32(16)])
interpreter.set_tensor(input_details[6]['index'], [np.float32(4)])
interpreter.set_tensor(input_details[7]['index'], [np.float32(1)])
interpreter.set_tensor(input_details[8]['index'], [np.float32(23)])
interpreter.set_tensor(input_details[9]['index'], [np.float32(0)])
interpreter.set_tensor(input_details[10]['index'], [np.float32(0)])


interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
