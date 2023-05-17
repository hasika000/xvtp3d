from typing import Any, Dict
import numpy as np


class Calibration_NS:  # todo:合并两个类
    """Calibration matrices and utils.

    3d XYZ are in 3D egovehicle coord.
    2d box xy are in image coord, normalized by width and height
    Point cloud are in egovehicle coord

    ::

       xy_image = K * [R|T] * xyz_ego

       xyz_image = [R|T] * xyz_ego

       image coord:
        ----> x-axis (u)
       |
       |
       v y-axis (v)

    egovehicle coord:
    front x, left y, up z
    """

    def __init__(self, intrinsic) -> None:
        self.K = intrinsic  # 这个值是不变的
        self.extrinsic = np.eye(4)  # 这个值是会变的

    def update_extrinsic(self, pos, rot, hight):
        # pos = [3] 世界坐标系下的xyz（实际上是标准化后的世界坐标系下的xyz）
        # rot = [2] 世界坐标系下的旋转
        # hight表示相机高度

        # 外参
        a = pos[0]
        b = pos[1]
        c = pos[2] + hight  # 在center的z方向上，hight，作为视线高度

        cosx = rot[0]
        sinx = -rot[1]  # 这里加负号是因为角度的方向变了

        T = np.array([[1., 0., 0., -a],
                      [0., 1., 0., -b],
                      [0., 0., 1., -c],
                      [0., 0., 0., 1.]])  # 平移矩阵，按照个体在世界坐标系的位置得出
        R1 = np.array([[cosx, -sinx, 0., 0.],
                       [sinx, cosx, 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])  # 按照个体的朝向求出的旋转矩阵
        R2 = np.array([[0., -1., 0., 0.],
                       [0., 0., -1., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 0., 1.]])  # 默认z轴为光轴，且平行于地平面

        world_2_cam = np.matmul(np.matmul(R2, R1), T)

        self.extrinsic = world_2_cam

    def cart2hom(self, pts_3d: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to Homogeneous.

        Args:
            pts_3d: nx3 points in Cartesian

        Returns:
            nx4 points in Homogeneous by appending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_ego_to_image(self, pts_3d_ego: np.ndarray) -> np.ndarray:
        """Project egovehicle coordinate to image.

        Args:
            pts_3d_ego: nx3 points in egovehicle coord

        Returns:
            nx3 points in image coord + depth
        """

        uv_cam = self.project_ego_to_cam(pts_3d_ego)
        return self.project_cam_to_image(uv_cam)

    def project_ego_to_cam(self, pts_3d_ego: np.ndarray) -> np.ndarray:
        """Project egovehicle point onto camera frame.

        Args:
            pts_3d_ego: nx3 points in egovehicle coord.

        Returns:
            nx3 points in camera coord.
        """

        uv_cam = self.extrinsic.dot(self.cart2hom(pts_3d_ego).transpose())

        return uv_cam.transpose()[:, 0:3]

    def project_cam_to_image(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Project camera coordinate to image.

        Args:
            pts_3d_ego: nx3 points in egovehicle coord

        Returns:
            nx3 points in image coord + depth
        """
        uv_cam = self.cart2hom(pts_3d_rect).T
        uv = self.K.dot(uv_cam)
        uv[0:2, :] /= uv[2, :]
        return uv.transpose()