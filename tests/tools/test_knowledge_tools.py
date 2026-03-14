"""Tests for knowledge_tools.py — MCP tools for knowledge graph operations.

Covers KGAddEntityTool, KGAddRelationTool, KGQueryTool, KGNeighborsTool
including spec properties, no-backend error paths, and successful execution
via a real KnowledgeGraphMemory instance.
"""

from __future__ import annotations

import json

import pytest

from openjarvis.tools.knowledge_tools import (
    KGAddEntityTool,
    KGAddRelationTool,
    KGNeighborsTool,
    KGQueryTool,
)
from openjarvis.tools.storage.knowledge_graph import (
    Entity,
    KnowledgeGraphMemory,
    Relation,
)

# ---------------------------------------------------------------------------
# Fixture: real KnowledgeGraphMemory backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg_backend(tmp_path):
    """Provide a temporary KnowledgeGraphMemory for tool tests."""
    backend = KnowledgeGraphMemory(db_path=tmp_path / "kg_tools_test.db")
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# KGAddEntityTool
# ---------------------------------------------------------------------------


class TestKGAddEntityTool:
    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_spec_name_and_category(self) -> None:
        tool = KGAddEntityTool()
        assert tool.spec.name == "kg_add_entity"
        assert tool.spec.category == "knowledge_graph"

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_spec_required_params(self) -> None:
        tool = KGAddEntityTool()
        required = tool.spec.parameters.get("required", [])
        assert "entity_id" in required
        assert "entity_type" in required
        assert "name" in required

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_spec_required_capabilities(self) -> None:
        tool = KGAddEntityTool()
        assert "memory:write" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_tool_id(self) -> None:
        assert KGAddEntityTool.tool_id == "kg_add_entity"

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_no_backend_returns_failure(self) -> None:
        tool = KGAddEntityTool(backend=None)
        result = tool.execute(
            entity_id="e1", entity_type="concept", name="Test",
        )
        assert result.success is False
        assert "No knowledge graph backend" in result.content

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_backend_without_add_entity_returns_failure(self) -> None:
        """A backend object that lacks add_entity is treated as missing."""

        class _Dummy:
            pass

        tool = KGAddEntityTool(backend=_Dummy())
        result = tool.execute(
            entity_id="e1", entity_type="concept", name="Test",
        )
        assert result.success is False

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_execute_success(self, kg_backend) -> None:
        tool = KGAddEntityTool(backend=kg_backend)
        result = tool.execute(
            entity_id="e1",
            entity_type="concept",
            name="Machine Learning",
            properties={"field": "AI"},
        )
        assert result.success is True
        assert "Machine Learning" in result.content
        # Verify entity was actually stored
        entity = kg_backend.get_entity("e1")
        assert entity is not None
        assert entity.name == "Machine Learning"
        assert entity.properties.get("field") == "AI"

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-entity")
    def test_execute_without_optional_properties(self, kg_backend) -> None:
        tool = KGAddEntityTool(backend=kg_backend)
        result = tool.execute(
            entity_id="e2", entity_type="tool", name="Calculator",
        )
        assert result.success is True
        entity = kg_backend.get_entity("e2")
        assert entity is not None
        assert entity.properties == {}


# ---------------------------------------------------------------------------
# KGAddRelationTool
# ---------------------------------------------------------------------------


class TestKGAddRelationTool:
    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_spec_name_and_category(self) -> None:
        tool = KGAddRelationTool()
        assert tool.spec.name == "kg_add_relation"
        assert tool.spec.category == "knowledge_graph"

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_spec_required_params(self) -> None:
        tool = KGAddRelationTool()
        required = tool.spec.parameters.get("required", [])
        assert "source_id" in required
        assert "target_id" in required
        assert "relation_type" in required

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_tool_id(self) -> None:
        assert KGAddRelationTool.tool_id == "kg_add_relation"

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_no_backend_returns_failure(self) -> None:
        tool = KGAddRelationTool(backend=None)
        result = tool.execute(
            source_id="a", target_id="b", relation_type="uses",
        )
        assert result.success is False
        assert "No knowledge graph backend" in result.content

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_backend_without_add_relation_returns_failure(self) -> None:
        class _Dummy:
            pass

        tool = KGAddRelationTool(backend=_Dummy())
        result = tool.execute(
            source_id="a", target_id="b", relation_type="uses",
        )
        assert result.success is False

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_execute_success(self, kg_backend) -> None:
        # Seed entities first
        kg_backend.add_entity(
            Entity(entity_id="a", entity_type="concept", name="A"),
        )
        kg_backend.add_entity(
            Entity(entity_id="b", entity_type="concept", name="B"),
        )
        tool = KGAddRelationTool(backend=kg_backend)
        result = tool.execute(
            source_id="a", target_id="b", relation_type="depends_on",
        )
        assert result.success is True
        assert "depends_on" in result.content
        assert kg_backend.relation_count() == 1

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_execute_with_custom_weight(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="x", entity_type="concept", name="X"),
        )
        kg_backend.add_entity(
            Entity(entity_id="y", entity_type="concept", name="Y"),
        )
        tool = KGAddRelationTool(backend=kg_backend)
        result = tool.execute(
            source_id="x", target_id="y",
            relation_type="similar_to", weight=0.75,
        )
        assert result.success is True

    @pytest.mark.spec("REQ-tools.knowledge-graph.add-relation")
    def test_execute_default_weight(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="p", entity_type="concept", name="P"),
        )
        kg_backend.add_entity(
            Entity(entity_id="q", entity_type="concept", name="Q"),
        )
        tool = KGAddRelationTool(backend=kg_backend)
        # No weight param — defaults to 1.0
        result = tool.execute(
            source_id="p", target_id="q", relation_type="uses",
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# KGQueryTool
# ---------------------------------------------------------------------------


class TestKGQueryTool:
    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_spec_name_and_category(self) -> None:
        tool = KGQueryTool()
        assert tool.spec.name == "kg_query"
        assert tool.spec.category == "knowledge_graph"

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_spec_required_capabilities(self) -> None:
        tool = KGQueryTool()
        assert "memory:read" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_tool_id(self) -> None:
        assert KGQueryTool.tool_id == "kg_query"

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_no_backend_returns_failure(self) -> None:
        tool = KGQueryTool(backend=None)
        result = tool.execute(entity_type="concept")
        assert result.success is False

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_backend_without_query_pattern(self) -> None:
        class _Dummy:
            pass

        tool = KGQueryTool(backend=_Dummy())
        result = tool.execute(entity_type="concept")
        assert result.success is False

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_execute_by_entity_type(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="t1", entity_type="tool", name="Calc"),
        )
        kg_backend.add_entity(
            Entity(entity_id="t2", entity_type="tool", name="Search"),
        )
        kg_backend.add_entity(
            Entity(entity_id="a1", entity_type="agent", name="Bot"),
        )
        tool = KGQueryTool(backend=kg_backend)
        result = tool.execute(entity_type="tool")
        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed["entities"]) == 2
        names = {e["name"] for e in parsed["entities"]}
        assert names == {"Calc", "Search"}

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_execute_by_relation_type(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="a", entity_type="x", name="A"),
        )
        kg_backend.add_entity(
            Entity(entity_id="b", entity_type="x", name="B"),
        )
        kg_backend.add_relation(
            Relation(source_id="a", target_id="b", relation_type="used"),
        )
        tool = KGQueryTool(backend=kg_backend)
        result = tool.execute(relation_type="used")
        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed["relations"]) == 1
        assert parsed["relations"][0]["type"] == "used"

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_execute_with_limit(self, kg_backend) -> None:
        for i in range(5):
            kg_backend.add_entity(
                Entity(entity_id=f"n{i}", entity_type="node", name=f"Node{i}"),
            )
        tool = KGQueryTool(backend=kg_backend)
        result = tool.execute(entity_type="node", limit=2)
        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed["entities"]) == 2

    @pytest.mark.spec("REQ-tools.knowledge-graph.query")
    def test_execute_default_limit(self, kg_backend) -> None:
        tool = KGQueryTool(backend=kg_backend)
        # No limit param — uses default 50
        result = tool.execute(entity_type="concept")
        assert result.success is True


# ---------------------------------------------------------------------------
# KGNeighborsTool
# ---------------------------------------------------------------------------


class TestKGNeighborsTool:
    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_spec_name_and_category(self) -> None:
        tool = KGNeighborsTool()
        assert tool.spec.name == "kg_neighbors"
        assert tool.spec.category == "knowledge_graph"

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_spec_required_params(self) -> None:
        tool = KGNeighborsTool()
        required = tool.spec.parameters.get("required", [])
        assert "entity_id" in required

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_tool_id(self) -> None:
        assert KGNeighborsTool.tool_id == "kg_neighbors"

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_no_backend_returns_failure(self) -> None:
        tool = KGNeighborsTool(backend=None)
        result = tool.execute(entity_id="e1")
        assert result.success is False

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_backend_without_neighbors_method(self) -> None:
        class _Dummy:
            pass

        tool = KGNeighborsTool(backend=_Dummy())
        result = tool.execute(entity_id="e1")
        assert result.success is False

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_execute_success(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="center", entity_type="concept", name="Center"),
        )
        kg_backend.add_entity(
            Entity(entity_id="n1", entity_type="concept", name="N1"),
        )
        kg_backend.add_entity(
            Entity(entity_id="n2", entity_type="concept", name="N2"),
        )
        kg_backend.add_relation(
            Relation(source_id="center", target_id="n1", relation_type="uses"),
        )
        kg_backend.add_relation(
            Relation(source_id="center", target_id="n2", relation_type="uses"),
        )
        tool = KGNeighborsTool(backend=kg_backend)
        result = tool.execute(entity_id="center")
        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed) == 2
        names = {e["name"] for e in parsed}
        assert names == {"N1", "N2"}

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_execute_with_relation_type_filter(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="a", entity_type="concept", name="A"),
        )
        kg_backend.add_entity(
            Entity(entity_id="b", entity_type="concept", name="B"),
        )
        kg_backend.add_entity(
            Entity(entity_id="c", entity_type="concept", name="C"),
        )
        kg_backend.add_relation(
            Relation(source_id="a", target_id="b", relation_type="uses"),
        )
        kg_backend.add_relation(
            Relation(source_id="a", target_id="c", relation_type="produces"),
        )
        tool = KGNeighborsTool(backend=kg_backend)
        result = tool.execute(
            entity_id="a", relation_type="uses", direction="out",
        )
        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "B"

    @pytest.mark.spec("REQ-tools.knowledge-graph.neighbors")
    def test_execute_with_direction_and_limit(self, kg_backend) -> None:
        kg_backend.add_entity(
            Entity(entity_id="hub", entity_type="concept", name="Hub"),
        )
        for i in range(5):
            nid = f"spoke{i}"
            kg_backend.add_entity(
                Entity(entity_id=nid, entity_type="concept", name=f"Spoke{i}"),
            )
            kg_backend.add_relation(
                Relation(source_id="hub", target_id=nid, relation_type="links"),
            )
        tool = KGNeighborsTool(backend=kg_backend)
        result = tool.execute(entity_id="hub", direction="out", limit=2)
        assert result.success is True
        parsed = json.loads(result.content)
        assert len(parsed) == 2
