import cv2
import numpy as np
import glob
import json

from tqdm import tqdm


class Calibration:
    def __init__(self):
        self.mtx = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.CHECKERBOARD = (6, 9)

    def video_calibrate(self, video_dir):
        cap = cv2.VideoCapture(video_dir)
        frames = []
        fms = []
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fms.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                frames.append(gray)
            except Exception:
                pass

        max_fm = np.argmax(fms)
        frames = np.asarray(frames)[fms > 1.5 * np.median(fms)]
        print(len(frames))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = []
        imgpoints = []

        objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        for i in tqdm(range(len(frames))):
            gray = frames[i]
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        cv2.destroyAllWindows()
        ret, self.mtx, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                      gray.shape[::-1], None, None)
        with open("calibration_videocamera.json", "w") as f:
            json.dump({'mtx': self.mtx.tolist(), 'dist_coeffs': self.dist_coeffs.tolist()}, f)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Total mean error for camera calibration: {mean_error / len(objpoints)}")

    def calibrate(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objpoints = []
        imgpoints = []

        objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        images = glob.glob('images/img_camera_calibration/*.jpg')
        for fname in images[:20]:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners2, ret)

            cv2.imshow('img', img)

        cv2.destroyAllWindows()
        ret, self.mtx, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                      gray.shape[::-1], None, None)

        print("Camera matrix : \n")
        print(self.mtx)
        print("dist_coeffs : \n")
        print(self.dist_coeffs)

        with open("calibration_camera.json", "w") as f:
            json.dump({'mtx': self.mtx.tolist(), 'dist_coeffs': self.dist_coeffs.tolist()}, f)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Total mean error for camera calibration: {mean_error / len(objpoints)}")
    
if __name__ == "__main__":
    calibration = Calibration()
    calibration.calibrate()
