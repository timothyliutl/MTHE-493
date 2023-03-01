from cosq_class import CoSQ
import numpy as np

obj = CoSQ(0.05, 2)
obj.training_set([np.random.randint(0,255) for i in range(100)])

print(obj.fit())

