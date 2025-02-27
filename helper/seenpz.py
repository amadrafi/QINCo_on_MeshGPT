import numpy as np


f = np.load("./shapenet/ShapeNetCore.v1.npz", allow_pickle=True)


for i in f['arr_0']:
  print(i['texts'])
