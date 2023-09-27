import tvm
from tvm import relay
import onnx
import json

# Load the ONNX model
name = "mlp_10_5_1"
onnx_model = onnx.load(f"{name}.onnx")

# Convert ONNX model to Relay IR
shape_dict = {"input": (1, 10)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Save Relay IR to JSON
relay_json = json.dumps(mod.astext())
with open(f"relay_ir_data/{name}_relay.json", "w") as f:
    f.write(relay_json)

# TVM optimizations and scheduling
target = "llvm"
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

# Convert the optimized Relay IR to Tensor IR (for simplicity, we'll just save the Relay IR after optimization)
optimized_relay_json = json.dumps(graph)
with open(f"tensor_ir_data/{name}_tensor.json", "w") as f:
    f.write(optimized_relay_json)
