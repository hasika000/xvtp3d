from typing import Any, Dict
import numpy as np


class Calibration:
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

    def __init__(self, calib: Dict[str, Any]) -> None:
        """Create a Calibration instance.

        Args:
            calib: Calibration data
        """

        self.calib_data = calib

        self.K = get_camera_intrinsic_matrix(calib["value"])  # 这个值是不变的
        self.extrinsic = np.eye(4)  # 这个值是会变的

        self.cu = self.calib_data["value"]["focal_center_x_px_"]
        self.cv = self.calib_data["value"]["focal_center_y_px_"]
        self.fu = self.calib_data["value"]["focal_length_x_px_"]
        self.fv = self.calib_data["value"]["focal_length_y_px_"]

        self.bx = self.K[0, 3] / (-self.fu)
        self.by = self.K[1, 3] / (-self.fv)

        self.camera = calib["key"][10:]

        self.fpv_type = 'none'

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

    def project_image_to_ego(self, uv_depth: np.ndarray) -> np.ndarray:
        """Project 2D image with depth to egovehicle coordinate.

        Args:
            uv_depth: nx3 first two channels are uv, 3rd channel
               is depth in camera coord. So basically in image coordinate.

        Returns:
            nx3 points in ego coord.
        """
        uv_cam = self.project_image_to_cam(uv_depth)
        return self.project_cam_to_ego(uv_cam)

    def project_image_to_cam(self, uv_depth: np.ndarray) -> np.ndarray:
        """Project 2D image with depth to camera coordinate.

        Args:
            uv_depth: nx3 first two channels are uv, 3rd channel
               is depth in camera coord.

        Returns:
            nx3 points in camera coord.
        """

        n = uv_depth.shape[0]

        x = ((uv_depth[:, 0] - self.cu) * uv_depth[:, 2]) / self.fu + self.bx
        y = ((uv_depth[:, 1] - self.cv) * uv_depth[:, 2]) / self.fv + self.by

        pts_3d_cam = np.zeros((n, 3))
        pts_3d_cam[:, 0] = x
        pts_3d_cam[:, 1] = y
        pts_3d_cam[:, 2] = uv_depth[:, 2]
        return pts_3d_cam

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

    def project_cam_to_ego(self, pts_3d_rect: np.ndarray) -> np.ndarray:
        """Project point in camera frame to egovehicle frame.

        Args:
            pts_3d_rect: nx3 points in cam coord.

        Returns:
            nx3 points in ego coord.
        """
        return np.linalg.inv((self.extrinsic)).dot(self.cart2hom(pts_3d_rect).transpose()).transpose()[:, 0:3]


def get_camera_intrinsic_matrix(camera_config: Dict[str, Any]) -> np.ndarray:
    """Load camera calibration data and constructs intrinsic matrix.

    Args:
       camera_config: Calibration config in json

    Returns:
       Camera intrinsic matrix.
    """
    intrinsic_matrix = np.zeros((3, 4))
    intrinsic_matrix[0, 0] = camera_config["focal_length_x_px_"]
    intrinsic_matrix[0, 1] = camera_config["skew_"]
    intrinsic_matrix[0, 2] = camera_config["focal_center_x_px_"]
    intrinsic_matrix[1, 1] = camera_config["focal_length_y_px_"]
    intrinsic_matrix[1, 2] = camera_config["focal_center_y_px_"]
    intrinsic_matrix[2, 2] = 1.0
    return intrinsic_matrix