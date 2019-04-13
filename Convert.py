import tensorflow as tf

converter = tf.contrib.lite.TFLiteConverter.from_saved_model("D:/Python Workspace/TFBussSchedule/model/1554839325")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
