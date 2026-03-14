import sys
from collections import Counter
import onnx

model_path = sys.argv[1] if len(sys.argv) > 1 else "mnist.onnx"
model = onnx.load(model_path)

counts = Counter(node.op_type for node in model.graph.node)
print(f"Operators in {model_path} ({sum(counts.values())} nodes, {len(counts)} unique):\n")
for op, count in counts.most_common():
    print(f"  {op}: {count}")

# Build Mermaid diagram
nodes = list(model.graph.node)
node_ids = [f"N{i}" for i in range(len(nodes))]

# Map each output tensor name to the node index that produces it
output_to_node = {}
for i, node in enumerate(nodes):
    for out in node.output:
        output_to_node[out] = i

graph_inputs = {inp.name for inp in model.graph.input}

print("\n```mermaid\nflowchart TD")

# Emit input node
print(f'  INPUT([Input])')
for i, node in enumerate(nodes):
    print(f'  {node_ids[i]}["{node.op_type}"]')

# Edges from graph input to first consuming nodes
for i, node in enumerate(nodes):
    for inp in node.input:
        if inp in graph_inputs:
            print(f'  INPUT --> {node_ids[i]}')

# Edges between nodes
for i, node in enumerate(nodes):
    for inp in node.input:
        if inp in output_to_node:
            src = output_to_node[inp]
            print(f'  {node_ids[src]} --> {node_ids[i]}')

# Edges to output
graph_output_names = {out.name for out in model.graph.output}
for i, node in enumerate(nodes):
    for out in node.output:
        if out in graph_output_names:
            print(f'  {node_ids[i]} --> OUTPUT([Output])')

print("```")
