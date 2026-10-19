import os
import cv2
import numpy as np

camera_cal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_cal")
image_list= []
rnd_tests_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_tests")

def calibrate_camera(images):
    checkerboard_size = (9, 6)

    object_point_template = np.zeros(
        (checkerboard_size[0] * checkerboard_size[1], 3),
        np.float32
    )

    object_point_template[:, :2] = np.mgrid[
        0:checkerboard_size[0],
        0:checkerboard_size[1]
    ].T.reshape(-1, 2)

    object_points = []
    image_points = []

    for gray_image in images:
        found, corners = cv2.findChessboardCorners(
            gray_image,
            checkerboard_size,
            None
        )

        if found:
            object_points.append(object_point_template)
            image_points.append(corners)
            print("Checkerboard detected")
        else:
            print("Checkerboard not detected")

    if len(object_points) == 0:
        raise RuntimeError(
            "No checkerboard corners were detected. "
            "Check checkerboard_size."
        )

    image_size = (
        images[0].shape[1],  # width
        images[0].shape[0]   # height
    )

    success, camera_matrix, distortion_coeffs, _, _ = \
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )

    if not success:
        raise RuntimeError("Camera calibration failed")

    return camera_matrix, distortion_coeffs

def undistort_image(image, camera_matrix, distortion_coeffs):
    return cv2.undistort(
        image,
        camera_matrix,
        distortion_coeffs,
        None,
        camera_matrix
    )

def main():
    print(camera_cal_dir)
    for file_name in os.listdir(camera_cal_dir):
        if file_name.lower().endswith((".jpg",".jpeg")):
                img_path = os.path.join(camera_cal_dir,file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
                image_list.append(img)

    # TEST FOR VISUALLY CHECKING IF IMAGES GET READ AS GRAYSCALE, WRITES THEM IN my_tests/
    #cnt = 1
    #for img in image_list:        
    #    cv2.imwrite(os.path.join(rnd_tests_output_dir, "img" + str(cnt) + ".jpg"), img)
    #    cnt += 1

    camera_matrix, distortion_coeffs = calibrate_camera(image_list)
    
    corrected = undistort_image(
    image_list[2],
    camera_matrix,
    distortion_coeffs
    )

    cv2.imwrite(
        os.path.join(rnd_tests_output_dir, "undistorted.jpg"),
        corrected
    )
    return


if __name__ == "__main__":
    main()