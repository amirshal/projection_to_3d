import cv2
import os
import numpy as np
from PIL import Image

class CameraInstance():
  def __init__(self,K,D=None):
    self.intrinsic=np.matrix(np.array(K,dtype = "float32").reshape(3,3))
    # convert for opencv

    self.fx = self.intrinsic[0, 0]
    self.fy = self.intrinsic[1, 1]
    self.cx = self.intrinsic[0, 2]
    self.cy = self.intrinsic[1, 2]

    self.distortion = np.array(D,dtype = "float32")

  def undistort_image(self,img):
    # undistort image
    return cv2.undistort(img, mtx, dist)


  def project(self,points):
    result = []
    rvec = tvec = np.array([0.0, 0.0, 0.0])
    if len(points) > 0:
      result, _ = cv2.projectPoints(points, rvec, tvec,
                                    self.intrinsic, self.distortion)
    return np.squeeze(result, axis=1)


  def _cv2_undistort_grid_points(self):
    w = 640
    h = 480
    uPoints, vPoints = np.meshgrid(range(0, w, 20), range(0, h, 20), indexing='xy')
    uPoints = uPoints.flatten()
    vPoints = vPoints.flatten()
    uvSrc = np.array([np.matrix([uPoints, vPoints]).transpose()], dtype="float32")
    assert uvSrc.shape[0] == 1
    assert uvSrc.shape[2] == 2
    uvDst = cv2.undistortPoints(uvSrc, self.intrinsic, self.distortion,None, self.intrinsic)
    uDst = [uv[0] for uv in uvDst[0]]
    vDst = [uv[1] for uv in uvDst[0]]
    return uDst, vDst

  def unproject_points(self,points, Z):
    # Step 1. Undistort.
    points_undistorted = np.array([])
    if len(points) > 0:
      points = np.expand_dims(points, axis=0)
      assert points.shape[0] == 1
      assert points.shape[2] == 2
      assert np.issubdtype(points.dtype, np.floating)
      self._cv2_undistort_grid_points()
      points_undistorted = cv2.undistortPoints(points, self.intrinsic, self.distortion, None, self.intrinsic )
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    # Step 2. Reproject.
    result = []
    for idx in range(points_undistorted.shape[0]):
      z = Z[0] if len(Z) == 1 else Z[idx]
      x = (points_undistorted[idx, 0] - self.cx) / self.fx * z
      y = (points_undistorted[idx, 1] - self.cy) / self.fy * z
      result.append([x, y, z])
    return result


def load_image(image_path, H, W, scale=True, resize=1):
    if image_path[-3:] == 'z16' or image_path[-3:] == 'z16':  # regular binary file
        with open(image_path, "rb") as f:
            fid = f.read(H * W *2)
            depth = np.frombuffer(fid, dtype=np.uint16).reshape(480, 640)
    return depth

if __name__ == '__main__':
    #intrinsic
    K = [383.37335205078125, 0.0, 309.12835693359375, 0.0, 382.4911804199219, 247.8705291748047,0.0,0.0,1.0]
    #distortion_model: plumb_bob
    d = [-0.05543722212314606, 0.06375012546777725, 0.00014851870946586132, -0.0005748926778323948, -0.02048298716545105]
    r = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
    p = [383.37335205078125, 0.0, 309.12835693359375, 0.0, 0.0, 382.4911804199219, 247.8705291748047,0.0,0.0,0.0,1.0,0.0]

    uv_points = np.array([[300,300],[400,400]],dtype = "float32")

    file_name = os.path.join(os.path.dirname(__file__), 'images', 'depth1.z16')
    depth_img = load_image(file_name,480,640)
    Z = [depth_img[int(a1)][int(a2)]/1000 for [a1,a2] in uv_points.tolist()]
    camera = CameraInstance(K,d)
    points_3d = camera.unproject_points(uv_points,Z)
    for i in range(len(uv_points)):
      print('2d point {} and z value {} will project to P:(dx:{:.2f}, dy:{:.2f}, dz:{:.2f})[m] '.format(uv_points[i],Z[i],points_3d[i][0],points_3d[i][1],points_3d[i][2]))