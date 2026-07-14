from __future__ import annotations

from typing import Any, Protocol

from ..graph import Graph
from .fusion import FusionPass


class GraphPass(Protocol):
    name: str

    def run(self, graph: Graph) -> dict[str, Any]:
        ...


class GraphPassPipeline:
    """
    Public C3.3 pipeline.

    Usage:
        optimized_graph, pass_results = GraphPassPipeline(
            enable_fusion=True
        ).run(graph)

    Required result path:
        pass_results["Fusion"]["stats"]["fusion_log"]
    """

    def __init__(
        self,
        *,
        enable_fusion: bool = True,
        validate_each_pass: bool = True,
        **_: Any,
    ) -> None:
        self.enable_fusion = enable_fusion
        self.validate_each_pass = validate_each_pass
        self.passes: list[GraphPass] = []
        self.pass_results: dict[str, Any] = {}

        if enable_fusion:
            self.passes.append(FusionPass())

    def add_pass(self, graph_pass: GraphPass) -> None:
        self.passes.append(graph_pass)

    def run(self, graph: Graph) -> tuple[Graph, dict[str, Any]]:
        optimized = graph.clone()
        self.pass_results = {}

        if self.validate_each_pass:
            optimized.validate()

        for graph_pass in self.passes:
            result = graph_pass.run(optimized)
            self.pass_results[graph_pass.name] = result
            if self.validate_each_pass:
                optimized.validate()

        return optimized, self.pass_results

    def __call__(self, graph: Graph) -> tuple[Graph, dict[str, Any]]:
        return self.run(graph)
