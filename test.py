import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import open3d as o3
import matplotlib.image as mpimg

rgb_file = 'rgb_1_0000001.png'
depth_file = 'depth_1_0000001.png'
color_raw = o3.read_image(rgb_file)
d1 = cv2.imread(depth_file)
ans = d1[:, :, 0] + d1[:, :, 1]
ans.astype(np.uint16)
depth_raw = o3.Image(ans)
rgbd_image = o3.create_rgbd_image_from_color_and_depth(color_raw, depth_raw)
print rgbd_image
plt.subplot(1, 3, 1)
plt.title('rgb image')
plt.imshow(color_raw)
plt.subplot(1, 3, 2)
plt.title('color image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 3, 3)
plt.title('depth image')
plt.imshow(rgbd_image.depth)
plt.show()

pcd = o3.create_point_cloud_from_rgbd_image(rgbd_image, o3.PinholeCameraIntrinsic(
    o3.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3.draw_geometries([pcd])
# image = np.concatenate([rgb, depth], axis=1)
# cv2.imshow('image', image)
key = cv2.waitKey(0)
key &= 255
if key == 27 or key == ord('q'):
    print("Pressed ESC or q, exiting")
    exit_request = True
else:
    exit_request = False
print 'ok'


def show_rgbd_image(image, depth_image, window_name='Image window', delay=1, depth_offset=0.0, depth_scale=1.0):
    if depth_image.dtype != np.uint8:
        if depth_scale is None:
            depth_scale = depth_image.max() - depth_image.min()
        if depth_offset is None:
            depth_offset = depth_image.min()
        depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
        depth_image = (255.0 * depth_image).astype(np.uint8)
    depth_image = np.tile(depth_image, (1, 1, 3))
    if image.shape[2] == 4:  # add alpha channel
        alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
        depth_image = np.concatenate([depth_image, alpha], axis=-1)
    images = np.concatenate([image, depth_image], axis=1)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow(window_name, images)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request

# joint_data_file = '/home/yufei/PycharmProjects/pose/joint_data.mat'
# prediction_file = '/home/yufei/PycharmProjects/pose/test_predictions.mat'
# test_file = '/home/yufei/PycharmProjects/pose/testing-2018-09-12_23_14_30.667727-result.txt'
# data = scio.loadmat(joint_data_file)
# data2 = scio.loadmat(prediction_file)
# frame_0_uvd = data['joint_uvd'][0][0]
# frame_0_xyz = data['joint_xyz'][0][0]
# plt.scatter(frame_0_uvd[:, 0], frame_0_uvd[:, 1])
# plt.scatter(frame_0_xyz[:, 0], frame_0_xyz[:, 1])
# plt.show()
# with open(test_file) as f:
#     test_data = f.readlines()
# for n, line in enumerate(test_data, 1):
#     line = line.split()
#     line.pop(0)
#     x = np.array(line)
#     x = x.astype(np.float64)
#     x = x.reshape(-1, 3)
#     # img = mpimg.imread('/home/yufei/PycharmProjects/pose/depth_1_0000001.png')
#     # imgplot = plt.imshow(img)
#     plt.scatter(x[:, 0], x[:, 1])
#
#     plt.show()
