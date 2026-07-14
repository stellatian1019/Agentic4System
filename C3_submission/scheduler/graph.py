from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .types import TensorInfo


@dataclass
class GraphNode:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)

    def copy(self, **updates: Any) -> "GraphNode":
        data = {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": dict(self.attributes),
        }
        data.update(updates)
        return GraphNode(**data)


@dataclass
class Graph:
    nodes: list[GraphNode]
    inputs: list[str]
    outputs: list[str]
    initializers: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)

    _producer: dict[str, GraphNode] = field(init=False, default_factory=dict)
    _consumers: dict[str, list[GraphNode]] = field(init=False, default_factory=dict)
    _node_by_name: dict[str, GraphNode] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.rebuild_index()

    @property
    def tensor_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        return {name: info.shape for name, info in self.tensors.items()}

    @property
    def tensor_dtypes(self) -> dict[str, str]:
        return {name: info.dtype for name, info in self.tensors.items()}

    def rebuild_index(self) -> None:
        self._producer = {}
        self._consumers = {}
        self._node_by_name = {}

        for node in self.nodes:
            if node.name in self._node_by_name:
                raise ValueError(f"Duplicate node name: {node.name}")
            self._node_by_name[node.name] = node

            for tensor_name in node.outputs:
                if not tensor_name:
                    continue
                if tensor_name in self._producer:
                    previous = self._producer[tensor_name]
                    raise ValueError(
                        f"Tensor {tensor_name!r} has two producers: "
                        f"{previous.name!r} and {node.name!r}"
                    )
                self._producer[tensor_name] = node

            for tensor_name in node.inputs:
                if tensor_name:
                    self._consumers.setdefault(tensor_name, []).append(node)

    def get_node(self, name: str) -> GraphNode | None:
        return self._node_by_name.get(name)

    def get_producer(self, tensor_name: str) -> GraphNode | None:
        return self._producer.get(tensor_name)

    def get_consumers(self, tensor_name: str) -> list[GraphNode]:
        return list(self._consumers.get(tensor_name, ()))

    def tensor_info(self, tensor_name: str) -> TensorInfo | None:
        return self.tensors.get(tensor_name)

    def tensor_shape(self, tensor_name: str):
        info = self.tensor_info(tensor_name)
        return None if info is None else info.shape

    def tensor_numel(self, tensor_name: str) -> int | None:
        info = self.tensor_info(tensor_name)
        return None if info is None else info.numel

    def node_index(self, node: GraphNode) -> int:
        for index, candidate in enumerate(self.nodes):
            if candidate is node or candidate.name == node.name:
                return index
        raise KeyError(f"Node {node.name!r} is not part of this graph")

    def topological_nodes(self) -> list[GraphNode]:
        indegree = {node.name: 0 for node in self.nodes}
        successors = {node.name: [] for node in self.nodes}

        for node in self.nodes:
            deps = set()
            for tensor_name in node.inputs:
                producer = self.get_producer(tensor_name)
                if producer is not None and producer.name != node.name:
                    deps.add(producer.name)
            indegree[node.name] = len(deps)
            for dependency in deps:
                successors[dependency].append(node.name)

        queue = [n.name for n in self.nodes if indegree[n.name] == 0]
        ordered = []

        while queue:
            current = queue.pop(0)
            ordered.append(self._node_by_name[current])
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)

        if len(ordered) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return ordered

    def validate(self) -> bool:
        available = set(self.inputs) | set(self.initializers)

        for node in self.topological_nodes():
            for tensor_name in node.inputs:
                if tensor_name and tensor_name not in available:
                    raise ValueError(
                        f"Node {node.name!r} references unavailable tensor "
                        f"{tensor_name!r}"
                    )
            available.update(name for name in node.outputs if name)

        for output_name in self.outputs:
            if output_name not in available:
                raise ValueError(f"Graph output {output_name!r} has no source")
        return True

    def clone(self) -> "Graph":
        return Graph(
            nodes=[node.copy() for node in self.nodes],
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            initializers=dict(self.initializers),
            tensors={
                name: TensorInfo(info.name, info.dtype, tuple(info.shape))
                for name, info in self.tensors.items()
            },
        )

    def replace_nodes(
        self,
        old_nodes: Iterable[GraphNode],
        new_node: GraphNode,
    ) -> None:
        old_names = {node.name for node in old_nodes}
        positions = [
            index for index, node in enumerate(self.nodes)
            if node.name in old_names
        ]
        if not positions:
            raise ValueError("No nodes selected for replacement")

        insertion_index = min(positions)
        remaining = [
            node for node in self.nodes if node.name not in old_names
        ]
        remaining.insert(insertion_index, new_node)
        self.nodes = remaining
        self.rebuild_index()
