import numpy as np
from arc_solver.src.attention.symbolic_attention import SymbolicAttention
from arc_solver.src.symbolic.rule_language import parse_rule


def test_attention_reranks():
    attn = SymbolicAttention(weight=1.0, dim=8)
    rule_a = [parse_rule("REPLACE [COLOR=0] -> [COLOR=1]")]
    rule_b = [parse_rule("TRANSLATE [COLOR=0] -> [COLOR=0]")]
    context = attn._embed_rule(rule_a)
    ranked = [(rule_b, 0.5), (rule_a, 0.5)]
    out = attn.apply(ranked, context)
    assert out[0][0] == rule_a
    assert out[0][1] > out[1][1]
