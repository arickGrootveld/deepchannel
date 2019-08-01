import numpy as np
from mismatch_data_gen import ARDatagenMismatch

trueState, measuredState = ARDatagenMismatch([10,2, 0.1,32, 20], 1)
print(trueState)
