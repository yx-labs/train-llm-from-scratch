import type { GraphEdge, GraphNode, GraphSpec, RuntimeError } from "../types";
import { ModuleRegistry } from "../moduleRegistry";

export type GraphValidationResult = {
  ok: boolean;
  errors: RuntimeError[];
};

export function validateGraphStructure(graph: GraphSpec, registry: ModuleRegistry): GraphValidationResult {
  const errors: RuntimeError[] = [];
  const nodesById = new Map<string, GraphNode>();

  graph.nodes.forEach((node) => {
    if (nodesById.has(node.id)) {
      errors.push({
        type: "graph_structure",
        message: `Duplicate node id: ${node.id}`,
        nodeId: node.id
      });
      return;
    }
    nodesById.set(node.id, node);
    if (!registry.maybeGet(node.moduleId)) {
      errors.push({
        type: "graph_structure",
        message: `Unknown module id: ${node.moduleId}`,
        nodeId: node.id
      });
    }
  });

  graph.edges.forEach((edge) => validateEdge(edge, nodesById, registry, errors));

  graph.nodes.forEach((node) => {
    const module = registry.maybeGet(node.moduleId);
    if (!module) return;
    module.inputs
      .filter((port) => port.required)
      .forEach((port) => {
        const connected = graph.edges.some((edge) => edge.to.nodeId === node.id && edge.to.portId === port.id);
        if (!connected) {
          errors.push({
            type: "missing_input",
            message: `Required input ${port.id} is not connected`,
            nodeId: node.id,
            portId: port.id,
            expected: port.accepts,
            suggestedProbe: `Connect a ${port.accepts?.join(" or ") ?? "value"} output to ${node.id}.${port.id}.`
          });
        }
      });
  });

  const cycleNodeId = findCycleNode(graph);
  if (cycleNodeId) {
    errors.push({
      type: "graph_structure",
      message: "Graph contains a cycle",
      nodeId: cycleNodeId
    });
  }

  return { ok: errors.length === 0, errors };
}

export function topologicalNodes(graph: GraphSpec) {
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]));
  const outgoing = new Map<string, string[]>();

  graph.edges.forEach((edge) => {
    indegree.set(edge.to.nodeId, (indegree.get(edge.to.nodeId) ?? 0) + 1);
    outgoing.set(edge.from.nodeId, [...(outgoing.get(edge.from.nodeId) ?? []), edge.to.nodeId]);
  });

  const queue = graph.nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0);
  const ordered: GraphNode[] = [];

  while (queue.length) {
    const node = queue.shift();
    if (!node) break;
    ordered.push(node);
    (outgoing.get(node.id) ?? []).forEach((targetId) => {
      const next = (indegree.get(targetId) ?? 0) - 1;
      indegree.set(targetId, next);
      if (next === 0) {
        const target = nodesById.get(targetId);
        if (target) queue.push(target);
      }
    });
  }

  return ordered;
}

function validateEdge(edge: GraphEdge, nodesById: Map<string, GraphNode>, registry: ModuleRegistry, errors: RuntimeError[]) {
  const fromNode = nodesById.get(edge.from.nodeId);
  const toNode = nodesById.get(edge.to.nodeId);
  if (!fromNode || !toNode) {
    errors.push({
      type: "graph_structure",
      message: `Edge ${edge.id} references a missing node`,
      nodeId: fromNode?.id ?? toNode?.id ?? "graph"
    });
    return;
  }

  const fromModule = registry.maybeGet(fromNode.moduleId);
  const toModule = registry.maybeGet(toNode.moduleId);
  const fromPort = fromModule?.outputs.find((port) => port.id === edge.from.portId);
  const toPort = toModule?.inputs.find((port) => port.id === edge.to.portId);
  if (!fromPort || !toPort) {
    errors.push({
      type: "graph_structure",
      message: `Edge ${edge.id} references an invalid port`,
      nodeId: !fromPort ? fromNode.id : toNode.id,
      portId: !fromPort ? edge.from.portId : edge.to.portId
    });
    return;
  }

  if (fromPort.emits && toPort.accepts && !toPort.accepts.includes(fromPort.emits)) {
    errors.push({
      type: "dtype_mismatch",
      message: `Port dtype mismatch from ${fromPort.emits} to ${toPort.accepts.join(" | ")}`,
      nodeId: toNode.id,
      portId: toPort.id,
      expected: toPort.accepts,
      received: fromPort.emits
    });
  }
}

function findCycleNode(graph: GraphSpec) {
  const ordered = topologicalNodes(graph);
  if (ordered.length === graph.nodes.length) return undefined;
  const orderedIds = new Set(ordered.map((node) => node.id));
  return graph.nodes.find((node) => !orderedIds.has(node.id))?.id;
}
