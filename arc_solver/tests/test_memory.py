from arc_solver.src.memory.lineage import RuleLineageTracker


def test_tracker_creation_and_insertion():
    tracker = RuleLineageTracker()
    tracker.add_entry("R1", parent_ids=[], source_task="000")
    assert tracker.export() == {"R1": {"parents": [], "source": "000"}}


def test_export_nested_ancestry():
    tracker = RuleLineageTracker()
    tracker.add_entry("R1", source_task="005", scoring_trace={"score": 0.92})
    tracker.add_entry("R2", parent_ids=["R1"], source_task="005", scoring_trace={"score": 0.77})
    tracker.add_entry("R3", parent_ids=["R2", "R1"], source_task="005", scoring_trace={"score": 0.85})
    data = tracker.export()
    assert data["R3"]["parents"] == ["R2", "R1"]
    assert data["R3"]["score"] == 0.85
    assert data["R1"]["score"] == 0.92
