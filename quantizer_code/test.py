from cosq_class import CoSQ
import numpy as np

obj = CoSQ(0.05, 1)
obj.training_set([np.random.randint(0,255) for i in range(100)])
obj.fit()
print(obj.centroids)
print(obj.centroid_map)
print(obj.quantize(200))

