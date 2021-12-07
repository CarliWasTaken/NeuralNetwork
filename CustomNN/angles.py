import numpy as np

# 200 left, 200 right, 200 forward
ANGLES = [0] * 400 + [1] * 400 + [2] * 400

TARGETS = np.zeros(3) + 0.01

# ANGLES = [a*100 for a in ANGLES]

# print(len(ANGLES))