# 이거 이해안됨... permutation은 리스트인데 우예 리스트 셔플을 하누
import numpy as np
import matplotlib.pyplot as plt
import math
m = 5
X = [[1,2,3],[4,5],[3,3,3],[4,4],[5,5]]
permutation = list(np.random.permutation(m))
shuffled_X = X[:, permutation]
print(shuffled_X)


