# Symbolic DSL Operators

## REPLACE

**Syntax:** `REPLACE(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## TRANSLATE

**Syntax:** `TRANSLATE(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## MERGE

**Syntax:** `MERGE(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## FILTER

**Syntax:** `FILTER(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## ROTATE

**Syntax:** `rotate_about_point(grid: 'Grid', center: 'Tuple[int, int]', angle: 'int') -> 'Grid'`

**Description:** Return ``grid`` rotated ``angle`` degrees about ``center``.

**Example:**

```python
# Example coming soon
```

## ROTATE90

**Syntax:** `ROTATE90(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## REFLECT

**Syntax:** `REFLECT(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## REPEAT

**Syntax:** `generate_repeat_composite_rules(input_grid: 'Grid', output_grid: 'Grid') -> 'List[CompositeRule]'`

**Description:** Return composite rules repeating ``input_grid`` then recolouring to match ``output_grid``.

**Example:**

```python
# Example coming soon
```

## SHAPE_ABSTRACT

**Syntax:** `SHAPE_ABSTRACT(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## CONDITIONAL

**Syntax:** `CONDITIONAL(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## REGION

**Syntax:** `REGION(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## FUNCTIONAL

**Syntax:** `FUNCTIONAL(...)`

**Description:** TODO: add description.

**Example:**

```python
# Example coming soon
```

## COMPOSITE

**Syntax:** `generate_repeat_composite_rules(input_grid: 'Grid', output_grid: 'Grid') -> 'List[CompositeRule]'`

**Description:** Return composite rules repeating ``input_grid`` then recolouring to match ``output_grid``.

**Example:**

```python
# Example coming soon
```

## dilate_zone

**DSL Keyword:** `dilate_zone`

**Transformation Type:** `FUNCTIONAL`

**Parameters:** zone_id

**DSL Version:** v1

**Description:** Dilate the pixels of ``zone_id`` by one cell inside ``zone_overlay``.

**Example:**

```python
dilate_zone(zone_id=1) [ZONE=1] -> [ZONE=1] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`

## draw_line

**DSL Keyword:** `draw_line`

**Transformation Type:** `FUNCTIONAL`

**Parameters:** p1, p2, color

**DSL Version:** v1

**Description:** Draw a 4-connected line on ``grid`` between ``point1`` and ``point2``.

**Example:**

```python
draw_line(p1=(0, 0), p2=(1, 1), color=1) [REGION=All] -> [REGION=All] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`

## erode_zone

**DSL Keyword:** `erode_zone`

**Transformation Type:** `FUNCTIONAL`

**Parameters:** zone_id

**DSL Version:** v1

**Description:** Erode ``zone_id`` by removing boundary pixels within ``zone_overlay``.

**Example:**

```python
erode_zone(zone_id=1) [ZONE=1] -> [ZONE=1] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`

## mirror_tile

**DSL Keyword:** `mirror_tile`

**Transformation Type:** `FUNCTIONAL`

**Parameters:** axis, repeats

**DSL Version:** v1

**Description:** Return grid tiled ``count`` times while mirroring every other tile.

**Example:**

```python
mirror_tile(axis=horizontal, repeats=2) [REGION=All] -> [REGION=All] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`

## pattern_fill

**DSL Keyword:** `pattern_fill`

**Transformation Type:** `FUNCTIONAL`

**Parameters:** mapping

**DSL Version:** v1

**Description:** Return ``grid`` with a pattern copied from ``source_zone_id`` to ``target_zone_id``.

**Example:**

```python
pattern_fill [REGION=All] -> [REGION=All] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`

## rotate_about_point

**DSL Keyword:** `rotate_about_point`

**Transformation Type:** `ROTATE`

**Parameters:** pivot, angle

**DSL Version:** v1

**Description:** Return ``grid`` rotated ``angle`` degrees about ``center``.

**Example:**

```python
rotate_about_point(pivot=(1,1), angle=90) [REGION=All] -> [REGION=All] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`

## zone_remap

**DSL Keyword:** `zone_remap`

**Transformation Type:** `FUNCTIONAL`

**Parameters:** mapping

**DSL Version:** v1

**Description:** Return a new grid with zones recoloured via ``zone_to_color``.

**Example:**

```python
zone_remap(mapping={1: 2}) [REGION=All] -> [REGION=All] [SPATIAL]
```

**Implementation:** `arc_solver/src/symbolic/vocabulary.py`
