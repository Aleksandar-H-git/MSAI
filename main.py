import os
import cv2
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))

camera_cal_dir = os.path.join(project_dir, "camera_cal")
test_images_dir = os.path.join(project_dir, "test_images")
rnd_tests_output_dir = os.path.join(project_dir, "my_tests")

image_list= []

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

    for file_name, gray_image in images:
        found, corners = cv2.findChessboardCorners(
            gray_image,
            checkerboard_size,
            None
        )

        if found:
            object_points.append(object_point_template)
            image_points.append(corners)
            print(f"Checkerboard detected: {file_name}")
        else:
            print(f"Checkerboard NOT detected: {file_name}")

    if len(object_points) == 0:
        raise RuntimeError("No checkerboard corners were detected.")

    image_size = (
        images[0][1].shape[1],
        images[0][1].shape[0]
    )

    _, camera_matrix, distortion_coeffs, _, _ = \
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )


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
    for file_name in os.listdir(camera_cal_dir):
        if file_name.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(camera_cal_dir, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                image_list.append((file_name, img))

    camera_matrix, distortion_coeffs = calibrate_camera(image_list)

    # testing undistorting test1.jpg
    test_image_path = os.path.join(
    test_images_dir,
    "test1.jpg"
    )

    img = cv2.imread(test_image_path)

    corrected = undistort_image(
    img,
    camera_matrix,
    distortion_coeffs
    )
    cv2.imwrite(
        os.path.join(rnd_tests_output_dir, "undistorted_test1.jpg"),
        corrected
    )

    print("Camera matrix:")
    print(camera_matrix)

    print("Distortion coefficients:")
    print(distortion_coeffs)


if __name__ == "__main__":
    main()