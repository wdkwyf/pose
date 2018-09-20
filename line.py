import open3d as o3
import numpy as np
import matplotlib.pyplot as plt

test_file = '/home/yufei/PycharmProjects/pose/testing-2018-09-12_23_14_30.667727-result.txt'
with open(test_file) as f:
    test_data = f.readlines()
for n, line in enumerate(test_data, 1):
    line = line.split()
    line.pop(0)
    x = np.array(line)
    x = x.astype(np.float64)
    x = x.reshape(-1, 3)
    # img = mpimg.imread('/home/yufei/PycharmProjects/pose/depth_1_0000001.png')
    # imgplot = plt.imshow(img)
    plt.scatter(x[:, 0], x[:, 1])

    plt.show()

print("Let\'s draw a cubic that consists of 8 points and 12 lines")
points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
          [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
lines = [[0, 1], [0, 2], [1, 3], [2, 3],
         [4, 5], [4, 6], [5, 7], [6, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3.LineSet()
line_set.points = o3.Vector3dVector(points)
line_set.lines = o3.Vector2iVector(lines)
line_set.colors = o3.Vector3dVector(colors)
o3.draw_geometries([line_set])
