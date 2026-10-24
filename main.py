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

def create_binary_image(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    absolute_sobel_x = np.absolute(sobel_x)

    maximum_gradient = np.max(absolute_sobel_x)
    if maximum_gradient == 0:
        scaled_sobel = np.zeros_like(gray)
    else:
        scaled_sobel = np.uint8(
            255 * absolute_sobel_x / maximum_gradient
        )

    gradient_binary = np.zeros_like(gray, dtype=np.uint8)
    gradient_binary[
        (scaled_sobel >= 20)
    ] = 1

    color_binary = np.zeros_like(gray, dtype=np.uint8)
    color_binary[
        (s_channel >= 100)
    ] = 1

    combined_binary = np.zeros_like(gray, dtype=np.uint8)
    combined_binary[
        (gradient_binary == 1) | (color_binary == 1)
    ] = 1

    return combined_binary

def perspective_transform(binary_image):
    height, width = binary_image.shape[:2]

    source_points = np.float32([
        [0.43 * width, 0.65 * height],  # top-left
        [0.58 * width, 0.65 * height],  # top-right
        [0.95 * width, 0.95 * height],  # bottom-right
        [0.10 * width, 0.95 * height]   # bottom-left
    ])

    destination_points = np.float32([
        [0.25 * width, 0],       # top-left
        [0.75 * width, 0],       # top-right
        [0.75 * width, height],  # bottom-right
        [0.25 * width, height]   # bottom-left
    ])

    transform_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )
    inverse_transform_matrix = cv2.getPerspectiveTransform(
        destination_points,
        source_points
    )

    warped = cv2.warpPerspective(
        binary_image,
        transform_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

    return warped, transform_matrix, inverse_transform_matrix

def find_lane_pixels(binary_warped, number_of_windows=9, margin=80, min_pixels=50):
    histogram = np.sum(
        binary_warped[binary_warped.shape[0] // 2:, :],
        axis=0
    )
    midpoint = histogram.shape[0] // 2
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    window_height = binary_warped.shape[0] // number_of_windows
    nonzero_y, nonzero_x = binary_warped.nonzero()
    left_current = left_base
    right_current = right_base
    left_lane_indices = []
    right_lane_indices = []

    debug_image = np.dstack(
        (binary_warped, binary_warped, binary_warped)
    ) * 255

    for window in range(number_of_windows):
        y_low = binary_warped.shape[0] - (window + 1) * window_height
        y_high = binary_warped.shape[0] - window * window_height
        left_x_low = left_current - margin
        left_x_high = left_current + margin
        right_x_low = right_current - margin
        right_x_high = right_current + margin

        cv2.rectangle(
            debug_image,
            (left_x_low, y_low),
            (left_x_high, y_high),
            (0, 255, 0),
            2
        )
        cv2.rectangle(
            debug_image,
            (right_x_low, y_low),
            (right_x_high, y_high),
            (0, 255, 0),
            2
        )

        left_indices = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= left_x_low)
            & (nonzero_x < left_x_high)
        ).nonzero()[0]
        right_indices = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= right_x_low)
            & (nonzero_x < right_x_high)
        ).nonzero()[0]

        left_lane_indices.append(left_indices)
        right_lane_indices.append(right_indices)

        if len(left_indices) > min_pixels:
            left_current = int(np.mean(nonzero_x[left_indices]))
        if len(right_indices) > min_pixels:
            right_current = int(np.mean(nonzero_x[right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    debug_image[left_y, left_x] = (255, 0, 0)
    debug_image[right_y, right_x] = (0, 0, 255)

    return left_x, left_y, right_x, right_y, debug_image

def main():
    for file_name in sorted(os.listdir(camera_cal_dir)):
        if file_name.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(camera_cal_dir, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                image_list.append((file_name, img))

    camera_matrix, distortion_coeffs = calibrate_camera(image_list)

    # Test the pipeline with a color road image from examples.
    test_image_path = os.path.join(test_images_dir, "challange00101.jpg")

    img = cv2.imread(test_image_path)

    if img is None:
        raise RuntimeError(f"Could not load image: {test_image_path}")

    corrected = undistort_image(
        img,
        camera_matrix,
        distortion_coeffs
    )

    binary = create_binary_image(corrected)
    warped_binary, _, inverse_transform_matrix = perspective_transform(binary)
    left_x, left_y, right_x, right_y, lane_pixels_image = \
        find_lane_pixels(warped_binary)

    os.makedirs(rnd_tests_output_dir, exist_ok=True)

    undistorted_path = os.path.join(
        rnd_tests_output_dir,
        "undistorted_example.jpg"
    )
    binary_path = os.path.join(
        rnd_tests_output_dir,
        "binary_example.jpg"
    )
    warped_path = os.path.join(
        rnd_tests_output_dir,
        "warped_binary_example.jpg"
    )
    lane_pixels_path = os.path.join(
        rnd_tests_output_dir,
        "lane_pixels_example.jpg"
    )

    if not cv2.imwrite(undistorted_path, corrected):
        raise RuntimeError(f"Could not save image: {undistorted_path}")
    # * 255 to see diff
    if not cv2.imwrite(binary_path, binary * 255):
        raise RuntimeError(f"Could not save image: {binary_path}")

    if not cv2.imwrite(warped_path, warped_binary * 255):
        raise RuntimeError(f"Could not save image: {warped_path}")

    if not cv2.imwrite(lane_pixels_path, lane_pixels_image):
        raise RuntimeError(f"Could not save image: {lane_pixels_path}")

    print(f"Saved undistorted image: {undistorted_path}")
    print(f"Saved binary image: {binary_path}")
    print(f"Saved warped binary image: {warped_path}")
    print(f"Saved lane pixel image: {lane_pixels_path}")
    print(f"Left lane pixels found: {len(left_x)}")
    print(f"Right lane pixels found: {len(right_x)}")

    print("Camera matrix:")
    print(camera_matrix)

    print("Distortion coefficients:")
    print(distortion_coeffs)


if __name__ == "__main__":
    main()
