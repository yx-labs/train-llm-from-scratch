import { ModuleRegistry } from "../moduleRegistry";
import type { GraphExecutionResult, GraphSpec, RuntimeValue, TensorShape, TraceFrame } from "../types";
import { valueKey } from "../types";
import { topologicalNodes, validateGraphStructure } from "./graphValidation";

export function executeGraph(graph: GraphSpec, registry: ModuleRegistry, testInputs: Record<string, RuntimeValue> = {}): GraphExecutionResult {
  const validation = validateGraphStructure(graph, registry);
  if (!validation.ok) {
    return {
      values: {},
      trace: [],
      error: validation.errors[0]
    };
  }

  const values: Record<string, RuntimeValue> = {};
  const trace: TraceFrame[] = [];
  const ordered = topologicalNodes(graph);

  for (const node of ordered) {
    const module = registry.get(node.moduleId);
    const inputs: Record<string, RuntimeValue> = {};
    graph.edges
      .filter((edge) => edge.to.nodeId === node.id)
      .forEach((edge) => {
        const value = values[valueKey(edge.from.nodeId, edge.from.portId)];
        if (value) inputs[edge.to.portId] = value;
      });

    const result = module.execute({ node, inputs, testInputs });
    const inputShapes = collectShapes(inputs);
    const outputShapes = collectShapes(result.outputs);
    const frame: TraceFrame = {
      step: trace.length,
      nodeId: node.id,
      moduleId: node.moduleId,
      inputShapes,
      outputShapes,
      samples: result.samples,
      error: result.error
    };
    trace.push(frame);

    if (result.error) {
      return { values, trace, error: result.error };
    }

    Object.entries(result.outputs).forEach(([portId, value]) => {
      if (value) values[valueKey(node.id, portId)] = value;
    });
  }

  return { values, trace };
}

function collectShapes(values: Partial<Record<string, RuntimeValue>>) {
  const shapes: Record<string, TensorShape> = {};
  Object.entries(values).forEach(([portId, value]) => {
    if (value?.shape) shapes[portId] = value.shape;
  });
  return shapes;
}
