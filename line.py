import open3d as o3
import numpy as np
import matplotlib.pyplot as plt

test_file = '/home/yufei/PycharmProjects/pose/testing-2018-09-12_23_14_30.667727-result.txt'
with open(test_file) as f:
    test_data = f.readlines()
for n, line in enumerate(test_data, 1):
    if n % 10 != 0:
        continue
    line = line.split()
    line.pop(0)
    x = np.array(line)
    x = x.astype(np.float64)
    x = x.reshape(-1, 3)
    x[:, 1] = -1 * x[:, 1]
    points = x
    plt.scatter(points[:, 0], x[:, 1])
    n = np.arange(14)
    for i, txt in enumerate(n):
        plt.annotate(txt, (points[i, 0], points[i, 1]))

    lines = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [11, 1], [12, 10], [13, 3], [13, 5], [13, 7], [13, 11], [13, 12],
             [10, 9], [11, 12]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3.LineSet()
    line_set.points = o3.Vector3dVector(points)
    line_set.lines = o3.Vector2iVector(lines)
    line_set.colors = o3.Vector3dVector(colors)
    o3.draw_geometries([line_set])
    # plt.show()
