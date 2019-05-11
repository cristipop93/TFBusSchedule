from tensorflow.contrib import predictor
import tensorflow as tf

# inputs = {
#     'idFrom': 'idFrom:0',
#     'idTo': 'idTo:0',
#     'vehicleType': 'vehicleType:0',
#     'month': 'month:0',
#     'day': 'day:0',
#     'hour': 'hour:0',
#     'minute': 'minute:0',
#     'holiday': 'holiday:0',
#     'vacation': 'vacation:0',
#     'temperature': 'temperature:0',
#     'pType': 'pType:0'
# }

predict_fn = predictor.from_saved_model("D:/Python Workspace/TFBussSchedule/model/1557588208", signature_def_key="predict")
                                        # input_names=inputs, output_names={'output': 'output'})


# def _int64_feature(value):
#     """Wrapper for inserting int64 features into Example proto."""
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


example = tf.train.Example(features=tf.train.Features(feature={
    'idFrom': _float_feature([5]),
    'idTo': _float_feature([8]),
    'vehicleType': _float_feature([1]),
    'month': _float_feature([1]),
    'day': _float_feature([6]),
    'hour': _float_feature([10]),
    'minute': _float_feature([24]),
    'holiday': _float_feature([0]),
    'vacation': _float_feature([0]),
    'temperature': _float_feature([-8]),
    'pType': _float_feature([0])
}))

example2 = {
    'idFrom': [1],
    'idTo': [8],
    'vehicleType': [0],
    'month': [4],
    'day': [5],
    'hour': [7],
    'minute': [16],
    'holiday': [0],
    'vacation': [0],
    'temperature': [12],
    'pType': [0]
}

# print(predict_fn({'inputs': [example.SerializeToString()]}))
print(predict_fn(input_dict=example2))
