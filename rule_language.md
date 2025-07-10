# Symbolic Rule DSL

## Overview

Symbolic rules define explicit transformations over ARC grids. Each rule encodes how certain colors, zones or regions are manipulated to morph an input grid into the desired output. They act as the building blocks of the solver – rules can be chained, validated and simulated step by step, enabling explainable reasoning rather than opaque pattern matching.

## Symbols

The vocabulary enumerates the atomic attributes available to rules:

```python
# symbolic/vocabulary.py
```
```python
class SymbolType(Enum):
    """Types of symbolic attributes."""
    COLOR = "COLOR"
    SHAPE = "SHAPE"
    REGION = "REGION"
    ZONE = "ZONE"
```

```python
class TransformationType(Enum):
    REPLACE = "REPLACE"
    TRANSLATE = "TRANSLATE"
    MERGE = "MERGE"
    FILTER = "FILTER"
    ROTATE = "ROTATE"
    REFLECT = "REFLECT"
    REPEAT = "REPEAT"
    CONDITIONAL = "CONDITIONAL"
    REGION = "REGION"
    FUNCTIONAL = "FUNCTIONAL"
    COMPOSITE = "COMPOSITE"
```

Each `Symbol` couples a type with a value and performs basic validity checks:

```python
@dataclass(frozen=True)
class Symbol:
    type: SymbolType
    value: str
    def __post_init__(self) -> None:
        if self.type is SymbolType.COLOR and not validate_color_range(self.value):
            raise ValueError(f"Invalid symbol value: {self.value}")
```

Transformations wrap a `TransformationType` and optional parameters:

```python
@dataclass(frozen=True)
class Transformation:
    ttype: TransformationType
    params: Dict[str, str] = field(default_factory=dict)
    def __str__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.ttype.value}({param_str})"
        return self.ttype.value
```

## SymbolicRule

A `SymbolicRule` specifies how a transformation acts on source symbols to produce target symbols. Additional metadata such as the nature of the rule (spatial, logical, symmetric) or a zone condition can be attached.

```python
@dataclass
class SymbolicRule:
    transformation: Transformation
    source: List[Symbol]
    target: List[Symbol]
    nature: TransformationNature | None = None
    condition: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
```
The string form of a rule mirrors the DSL and is used for serialization:

```python
left = ", ".join(str(s) for s in self.source)
right = ", ".join(str(s) for s in self.target)
return f"{self.transformation.ttype.value} [{left}] -> [{right}]"
```

## CompositeRule

Complex behaviours are represented by chaining multiple `SymbolicRule` instances in a `CompositeRule`. Each step is applied in sequence when simulating the rule.

```python
@dataclass
class CompositeRule:
    steps: List[SymbolicRule]
    nature: TransformationNature | None = None
    meta: Dict[str, Any] = field(default_factory=dict)
    transformation: Transformation = field(default_factory=lambda: Transformation(TransformationType.COMPOSITE))

    def simulate(self, grid: Grid) -> Grid:
        out = grid
        for step in self.steps:
            out = safe_apply_rule(step, out, perform_checks=False)
        return out
```

Conditions bind rules to spatial constraints. For example, `validate_rule_application` checks that a requested `zone` exists before the rule can be used:

```python
zone = rule.condition.get("zone") if rule.condition else None
if zone:
    overlay = zone_overlay(grid)
    zones = {sym.value for row in overlay for sym in row if sym}
    if zone not in zones:
        return False
```

## Serialisation and Parsing

Rules are written using a compact DSL. `parse_rule` converts a string like `"REPLACE [COLOR=1] -> [COLOR=2]"` into a `SymbolicRule` object. `rules_to_program` and `program_to_rules` serialize and parse whole programs using `|` as a separator.

```python
rule = parse_rule("REPLACE [COLOR=1] -> [COLOR=2]")
program = rules_to_program([rule])
recovered = program_to_rules(program)
```

## Simulation and Validation

When executing programs the simulator first filters out impossible rules with `validate_color_dependencies` and then applies each rule using `safe_apply_rule`. Composite chains are expanded and simulated step by step. Reflex and training constraints are enforced during `_safe_apply_rule`.

```python
validated = validate_color_dependencies([comp], grid)
result = safe_apply_rule(comp, grid)
```

### Example

```yaml
# simple replace
- transformation: REPLACE
  source:
    - type: COLOR
      value: "1"
  target:
    - type: COLOR
      value: "2"

# composite example
- transformation: COMPOSITE
  steps:
    - {transformation: REPLACE, source: [ {type: COLOR, value: "1"} ], target: [ {type: COLOR, value: "2"} ] }
    - {transformation: REPLACE, source: [ {type: COLOR, value: "2"} ], target: [ {type: COLOR, value: "3"} ] }
```

## Limitations and Future Work

The rule language currently focuses on colour and region manipulations. Shape abstraction and robust rotation handling are planned but not yet implemented. Composite growth is checked via `grid_growth_forecast()` to keep the uncertainty map within bounds.

