from tensorflow.contrib import predictor

predict_fn = predictor.from_saved_model("D:/Python Workspace/TFBussSchedule/model/1554230641")
# pred = predict_fn(
#     {"inputs": [0, 1, 2, 1, 6, 10, 24, 0, 0, -8, 0]
#      })['output']
# print(pred)

pred = predict_fn({'inputs': {'idFrom': [0],
                   'idTo': [1],
                   'vehicleType': [2],
                   'month': [1],
                   'day': [6],
                   'hour': [10],
                   'minute': [24],
                   'holiday': [0],
                   'vacation': [0],
                   'temperature': [-8],
                   'pType': [0]
                   }})
#
# inputs = {'idFrom': 0,
#           'idTo': 1,
#           'vehicleType': 2,
#           'month': 1,
#           'day': 6,
#           'hour': 10,
#           'minute': 24,
#           'holiday': 0,
#           'vacation': 0,
#           'temperature': -8,
#           'pType': 0
#           }
#
# pred = predict_fn(inputs)

print(pred)
