import tensorflow as tf

MODEL = "global_water_quality_idcnn.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()

print("\n========== INPUT ==========")
for x in inputs:
    print("name   :", x["name"])
    print("shape  :", x["shape"])
    print("dtype  :", x["dtype"])
    print("index  :", x["index"])
    print("quant  :", x["quantization"])

print("\n========== OUTPUT ==========")
for x in outputs:
    print("name   :", x["name"])
    print("shape  :", x["shape"])
    print("dtype  :", x["dtype"])
    print("index  :", x["index"])
    print("quant  :", x["quantization"])

print("\nTFLite model loaded successfully.")