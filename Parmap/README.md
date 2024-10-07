# Parmap
Parmap is a versatile Python framework designed to support parallel execution of code via a map-reduce paradigm. 

## Installation 
- You can install parmap and its dependencies simply by cloning the repo and running:
```pip install -e .```

## Example

```
from parmap.parmap import Parmap  

def cubed(i):
    from math import pow
    return pow(i, 3)

parmap = Parmap(mode="seq")
parmap(fn=cubed, items=[1, 2 , 3, 4])
```
