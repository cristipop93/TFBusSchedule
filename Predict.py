from tensorflow.contrib import predictor
import tensorflow as tf

predict_fn = predictor.from_saved_model("D:/Python Workspace/TFBussSchedule/model/1554499492")


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
    'idFrom': _float_feature([0]),
    'idTo': _float_feature([1]),
    'vehicleType': _float_feature([2]),
    'month': _float_feature([1]),
    'day': _float_feature([6]),
    'hour': _float_feature([10]),
    'minute': _float_feature([24]),
    'holiday': _float_feature([0]),
    'vacation': _float_feature([0]),
    'temperature': _float_feature([-8]),
    'pType': _float_feature([0])
}))

example2 = tf.train.Example(features=tf.train.Features(feature={
    'idFrom': _float_feature([0]),
    'idTo': _float_feature([1]),
    'vehicleType': _float_feature([0]),
    'month': _float_feature([4]),
    'day': _float_feature([5]),
    'hour': _float_feature([11]),
    'minute': _float_feature([16]),
    'holiday': _float_feature([1]),
    'vacation': _float_feature([0]),
    'temperature': _float_feature([23]),
    'pType': _float_feature([1])
}))

print(predict_fn({'inputs': [example.SerializeToString()]}))
print(predict_fn({'inputs': [example2.SerializeToString()]}))
