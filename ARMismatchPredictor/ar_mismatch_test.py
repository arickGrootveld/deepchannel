import numpy as np
from mismatch_data_gen import ARDatagenMismatch

trueState, measuredState = ARDatagenMismatch([2,2, 0.1,10, 20], 1)
print(trueState)
