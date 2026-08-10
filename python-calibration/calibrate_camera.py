import cv2
import numpy as np
import glob

# --- CONFIGURATION ---
# OpenCV counts the INNER corners of the checkerboard. 
# A 10x7 square grid has 9x6 inner intersections.
CHECKERBOARD = (9, 6)

# The directory where you placed your 15-20 phone photos
IMAGE_DIR = "calibration_images/*.jpg"
# ---------------------

# Sub-pixel optimization criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

images = glob.glob(IMAGE_DIR)

if not images:
    print(f"No images found matching '{IMAGE_DIR}'.")
    exit()

print(f"Found {len(images)} images. Processing...")

gray = None
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # Refine corner locations to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"[{fname}] Corners found!")
    else:
        print(f"[{fname}] Corners NOT found. (Try taking the photo closer or checking lighting).")

if not objpoints:
    print("Failed to find checkerboard corners in any images. Exiting.")
    exit()

print("\nCalibrating camera... (this might take a few seconds)")
# Perform calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n" + "="*60)
print("CALIBRATION SUCCESSFUL! Copy-paste this into KeyboardMap.cpp:")
print("="*60 + "\n")

print("// Replace the dummy matrix inside updateTransform() with this:")
print(f"camera_matrix_ = (cv::Mat_<double>(3, 3) << ")
print(f"    {mtx[0][0]:.5f}, {mtx[0][1]:.5f}, {mtx[0][2]:.5f},")
print(f"    {mtx[1][0]:.5f}, {mtx[1][1]:.5f}, {mtx[1][2]:.5f},")
print(f"    {mtx[2][0]:.5f}, {mtx[2][1]:.5f}, {mtx[2][2]:.5f});")

print("\n// Lens distortion coefficients")
dist_flat = dist.flatten()
print(f"dist_coeffs_ = (cv::Mat_<double>(5, 1) << ")
print(f"    {dist_flat[0]:.5f}, {dist_flat[1]:.5f}, {dist_flat[2]:.5f}, {dist_flat[3]:.5f}, {dist_flat[4]:.5f});")
print("\n" + "="*60)