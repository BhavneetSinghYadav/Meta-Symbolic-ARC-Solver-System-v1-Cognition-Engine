from arc_solver.src.segmentation.segmenter import segment

def test_segment_returns_list():
    assert isinstance(segment([]), list)
