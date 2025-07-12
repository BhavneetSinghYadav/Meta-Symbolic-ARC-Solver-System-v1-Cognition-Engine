# GridProxy

`GridProxy` is a lightweight wrapper around a numpy array that caches commonly used
information derived from the grid.  It stores the zone overlay, colour entropy and
basic shape descriptors so that subsequent accesses do not require recomputation.
The underlying numpy data should never be mutated once wrapped.

## Benefits

* Centralises zone segmentation logic.
* Reuses entropy and segmentation data across simulation and scoring.
* Reduces repeated calls to expensive segmentation functions.

## Usage

```python
import numpy as np
from arc_solver.src.core.grid_proxy import GridProxy

arr = np.array([[1, 2], [3, 4]])
proxy = GridProxy(arr)
entropy = proxy.get_entropy()         # cached
overlay = proxy.get_zone_overlay()    # cached
shapes = proxy.get_shapes()
```

Avoid serialising `GridProxy` objects since their caches are lazy and hold
references to numpy arrays.
