import onnx

def load_onnx_model(file_path):
    return onnx.load(file_path)

def inspect_onnx_model(model):
    # Iterate through each node in the computational graph
    for node in model.graph.node:
        print("Operation:", node.op_type)
        print("Inputs:", [inp for inp in node.input])
        print("Outputs:", [out for out in node.output])
