import os
import cv2
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))

camera_cal_dir = os.path.join(project_dir, "camera_cal")
test_images_dir = os.path.join(project_dir, "test_images")
test_videos_dir = os.path.join(project_dir, "test_videos")
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
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
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
        (scaled_sobel >= 40) & (scaled_sobel <= 180)
    ] = 1

    white_mask = (
        (l_channel >= 210) &
        (s_channel <= 100)
    )
    yellow_mask = (
        (h_channel >= 15) &
        (h_channel <= 35) &
        (l_channel >= 100) &
        (s_channel >= 100)
    )

    color_binary = np.zeros_like(gray, dtype=np.uint8)
    color_binary[white_mask | yellow_mask] = 1

    combined_binary = np.zeros_like(gray, dtype=np.uint8)
    combined_binary[
        (color_binary == 1) |
        ((gradient_binary == 1) & (l_channel >= 160))
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

def find_lane_pixels(binary_warped, number_of_windows=9, margin=70, min_pixels=50):
    histogram = np.sum(
        binary_warped[binary_warped.shape[0] // 2:, :],
        axis=0
    )
    midpoint = histogram.shape[0] // 2
    width = histogram.shape[0]

    left_min = int(width * 0.10)
    left_max = int(width * 0.50)
    right_min = int(width * 0.50)
    right_max = int(width * 0.90)

    left_base = int(
        np.argmax(histogram[left_min:left_max]) + left_min
    )
    right_base = int(
        np.argmax(histogram[right_min:right_max]) + right_min
    )
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

def find_lane_pixels_near_fit(
    binary_warped,
    left_fit,
    right_fit,
    margin=80
):
    nonzero_y, nonzero_x = binary_warped.nonzero()
    left_position = np.polyval(left_fit, nonzero_y)
    right_position = np.polyval(right_fit, nonzero_y)

    left_indices = (
        (nonzero_x >= left_position - margin) &
        (nonzero_x <= left_position + margin)
    )
    right_indices = (
        (nonzero_x >= right_position - margin) &
        (nonzero_x <= right_position + margin)
    )

    left_x = nonzero_x[left_indices]
    left_y = nonzero_y[left_indices]
    right_x = nonzero_x[right_indices]
    right_y = nonzero_y[right_indices]

    debug_image = np.dstack(
        (binary_warped, binary_warped, binary_warped)
    ) * 255
    debug_image[left_y, left_x] = (255, 0, 0)
    debug_image[right_y, right_x] = (0, 0, 255)

    return left_x, left_y, right_x, right_y, debug_image

def fit_lane_polynomials(left_x, left_y, right_x, right_y, image_height):
    if len(left_x) < 200 or len(right_x) < 200:
        raise ValueError(
            f"Not enough lane pixels: left={len(left_x)}, right={len(right_x)}"
        )

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    plot_y = np.linspace(0, image_height - 1, image_height)
    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)

    return left_fit, right_fit, plot_y, left_fit_x, right_fit_x

def calculate_curvature_and_offset(left_fit, right_fit, image_shape):
    height, width = image_shape[:2]
    plot_y = np.linspace(0, height - 1, height)
    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)

    lane_width_pixels = right_fit_x[-1] - left_fit_x[-1]
    if lane_width_pixels <= 0:
        raise ValueError("The detected lane width is invalid.")

    y_meters_per_pixel = 30 / height
    x_meters_per_pixel = 3.7 / lane_width_pixels
    y_in_meters = plot_y * y_meters_per_pixel

    left_fit_meters = np.polyfit(
        y_in_meters,
        left_fit_x * x_meters_per_pixel,
        2
    )
    right_fit_meters = np.polyfit(
        y_in_meters,
        right_fit_x * x_meters_per_pixel,
        2
    )
    y_evaluation = (height - 1) * y_meters_per_pixel

    def curvature(fit):
        denominator = abs(2 * fit[0])
        if denominator < 1e-9:
            return float("inf")
        return (
            (1 + (2 * fit[0] * y_evaluation + fit[1]) ** 2) ** 1.5
        ) / denominator

    left_curvature = curvature(left_fit_meters)
    right_curvature = curvature(right_fit_meters)
    average_curvature = (left_curvature + right_curvature) / 2

    lane_center = (left_fit_x[-1] + right_fit_x[-1]) / 2
    image_center = width / 2
    vehicle_offset = (
        image_center - lane_center
    ) * x_meters_per_pixel

    return average_curvature, vehicle_offset

def validate_lane_fit(
    left_fit,
    right_fit,
    image_shape,
    previous_fits=None
):
    height, width = image_shape[:2]
    check_y = np.array([0, height - 1])
    left_x = np.polyval(left_fit, check_y)
    right_x = np.polyval(right_fit, check_y)
    lane_widths = right_x - left_x

    if np.any(lane_widths < width * 0.20) or \
            np.any(lane_widths > width * 0.80):
        raise ValueError("Detected lane width is unrealistic.")

    if abs(lane_widths[1] - lane_widths[0]) > width * 0.25:
        raise ValueError("Lane boundaries are not approximately parallel.")

    lane_center = (left_x[-1] + right_x[-1]) / 2
    if abs(lane_center - width / 2) > width * 0.35:
        raise ValueError("Detected lane position is unrealistic.")

    if previous_fits is not None:
        previous_left, previous_right = previous_fits
        previous_center = (
            np.polyval(previous_left, height - 1) +
            np.polyval(previous_right, height - 1)
        ) / 2
        if abs(lane_center - previous_center) > width * 0.15:
            raise ValueError("Lane position changed too quickly.")

def draw_lane_overlay(
    undistorted,
    binary_warped,
    left_fit,
    right_fit,
    inverse_transform_matrix,
    curvature,
    vehicle_offset
):
    height, width = binary_warped.shape[:2]
    plot_y = np.linspace(0, height - 1, height)
    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)

    color_warp = np.zeros((height, width, 3), dtype=np.uint8)
    left_points = np.column_stack((left_fit_x, plot_y))
    right_points = np.flipud(
        np.column_stack((right_fit_x, plot_y))
    )
    lane_points = np.vstack((left_points, right_points)).astype(np.int32)
    cv2.fillPoly(color_warp, [lane_points], (0, 255, 0))

    road_overlay = cv2.warpPerspective(
        color_warp,
        inverse_transform_matrix,
        (width, height)
    )
    result = cv2.addWeighted(undistorted, 1, road_overlay, 0.3, 0)

    cv2.putText(
        result,
        f"Lane curvature: {curvature:.0f} m",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    side = "right" if vehicle_offset > 0 else "left"
    cv2.putText(
        result,
        f"Vehicle offset: {abs(vehicle_offset):.2f} m {side} of center",
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return result

class LaneTracker:
    def __init__(self, history_size=5, maximum_reuse_frames=5):
        self.history_size = history_size
        self.maximum_reuse_frames = maximum_reuse_frames
        self.left_fits = []
        self.right_fits = []
        self.failed_frames = 0

    def has_fit(self):
        return bool(self.left_fits and self.right_fits)

    def current_fits(self):
        return (
            np.mean(self.left_fits, axis=0),
            np.mean(self.right_fits, axis=0)
        )

    def can_reuse(self):
        return (
            self.has_fit() and
            self.failed_frames < self.maximum_reuse_frames
        )

    def mark_failure(self):
        self.failed_frames += 1

    def update(self, left_fit, right_fit):
        self.left_fits.append(left_fit)
        self.right_fits.append(right_fit)
        self.left_fits = self.left_fits[-self.history_size:]
        self.right_fits = self.right_fits[-self.history_size:]
        self.failed_frames = 0

        return self.current_fits()

def process_frame(
    image,
    camera_matrix,
    distortion_coeffs,
    tracker=None,
    return_steps=False
):
    undistorted = undistort_image(
        image,
        camera_matrix,
        distortion_coeffs
    )
    binary = create_binary_image(undistorted)
    warped, _, inverse_matrix = perspective_transform(binary)

    previous_fits = None
    if tracker is not None and tracker.can_reuse():
        previous_fits = tracker.current_fits()

    try:
        if previous_fits is None:
            lane_pixels = find_lane_pixels(warped)
        else:
            lane_pixels = find_lane_pixels_near_fit(
                warped,
                previous_fits[0],
                previous_fits[1]
            )

        left_x, left_y, right_x, right_y, lane_pixel_image = lane_pixels
        left_fit, right_fit, _, _, _ = fit_lane_polynomials(
            left_x,
            left_y,
            right_x,
            right_y,
            warped.shape[0]
        )
        validate_lane_fit(
            left_fit,
            right_fit,
            warped.shape,
            previous_fits
        )
        curvature, vehicle_offset = calculate_curvature_and_offset(
            left_fit,
            right_fit,
            warped.shape
        )
        if np.isnan(curvature) or curvature < 20:
            raise ValueError("Detected lane curvature is unrealistic.")
        if not np.isfinite(vehicle_offset) or abs(vehicle_offset) > 2.5:
            raise ValueError("Detected vehicle position is unrealistic.")

        if tracker is not None:
            left_fit, right_fit = tracker.update(left_fit, right_fit)
            curvature, vehicle_offset = calculate_curvature_and_offset(
                left_fit,
                right_fit,
                warped.shape
            )
    except ValueError:
        if tracker is None or not tracker.can_reuse():
            raise

        tracker.mark_failure()
        left_fit, right_fit = tracker.current_fits()
        curvature, vehicle_offset = calculate_curvature_and_offset(
            left_fit,
            right_fit,
            warped.shape
        )
        lane_pixel_image = np.dstack((warped, warped, warped)) * 255

    result = draw_lane_overlay(
        undistorted,
        warped,
        left_fit,
        right_fit,
        inverse_matrix,
        curvature,
        vehicle_offset
    )

    if return_steps:
        return result, {
            "undistorted": undistorted,
            "binary": binary,
            "warped": warped,
            "lane_pixels": lane_pixel_image
        }

    return result

def process_video(
    input_path,
    output_path,
    camera_matrix,
    distortion_coeffs,
    maximum_frames=None
):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    tracker = LaneTracker()
    frame_count = 0
    failed_frames = 0

    while maximum_frames is None or frame_count < maximum_frames:
        success, frame = capture.read()
        if not success:
            break

        try:
            result = process_frame(
                frame,
                camera_matrix,
                distortion_coeffs,
                tracker=tracker
            )
        except ValueError:
            failed_frames += 1
            result = undistort_image(
                frame,
                camera_matrix,
                distortion_coeffs
            )

        writer.write(result)
        frame_count += 1

    capture.release()
    writer.release()
    print(
        f"Saved video: {output_path} "
        f"({frame_count} frames, {failed_frames} failed)"
    )

def main():
    for file_name in sorted(os.listdir(camera_cal_dir)):
        if file_name.lower().endswith((".jpg", ".jpeg")):
            img_path = os.path.join(camera_cal_dir, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                image_list.append((file_name, img))

    camera_matrix, distortion_coeffs = calibrate_camera(image_list)

    os.makedirs(rnd_tests_output_dir, exist_ok=True)

    test_image_path = os.path.join(test_images_dir, "custom_test2.jpg")

    img = cv2.imread(test_image_path)

    if img is None:
        raise RuntimeError(f"Could not load image: {test_image_path}")

    img = cv2.resize(img, (1280, 720))

    corrected = undistort_image(
        img,
        camera_matrix,
        distortion_coeffs
    )
    original_vs_undistorted = np.hstack((img, corrected))

    binary = create_binary_image(corrected)
    warped_binary, _, inverse_transform_matrix = perspective_transform(binary)
    left_x, left_y, right_x, right_y, lane_pixels_image = \
        find_lane_pixels(warped_binary)
    left_fit, right_fit, plot_y, left_fit_x, right_fit_x = \
        fit_lane_polynomials(
            left_x,
            left_y,
            right_x,
            right_y,
            warped_binary.shape[0]
        )

    fitted_lane_image = lane_pixels_image.copy()
    image_width = fitted_lane_image.shape[1]
    left_points = np.column_stack((
        np.clip(left_fit_x, 0, image_width - 1),
        plot_y
    )).astype(np.int32)
    right_points = np.column_stack((
        np.clip(right_fit_x, 0, image_width - 1),
        plot_y
    )).astype(np.int32)
    cv2.polylines(
        fitted_lane_image,
        [left_points],
        False,
        (0, 255, 255),
        4
    )

    curvature, vehicle_offset = calculate_curvature_and_offset(
        left_fit,
        right_fit,
        warped_binary.shape
    )
    final_lane_image = draw_lane_overlay(
        corrected,
        warped_binary,
        left_fit,
        right_fit,
        inverse_transform_matrix,
        curvature,
        vehicle_offset
    )
    cv2.polylines(
        fitted_lane_image,
        [right_points],
        False,
        (0, 255, 255),
        4
    )

    undistorted_path = os.path.join(
        rnd_tests_output_dir,
        "undistorted_example.jpg"
    )
    comparison_path = os.path.join(
        rnd_tests_output_dir,
        "original_vs_undistorted_example.jpg"
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
    fitted_lane_path = os.path.join(
        rnd_tests_output_dir,
        "fitted_lane_example.jpg"
    )
    final_lane_path = os.path.join(
        rnd_tests_output_dir,
        "final_lane_example.jpg"
    )

    if not cv2.imwrite(undistorted_path, corrected):
        raise RuntimeError(f"Could not save image: {undistorted_path}")

    if not cv2.imwrite(comparison_path, original_vs_undistorted):
        raise RuntimeError(f"Could not save image: {comparison_path}")

    # * 255 to see diff
    if not cv2.imwrite(binary_path, binary * 255):
        raise RuntimeError(f"Could not save image: {binary_path}")

    if not cv2.imwrite(warped_path, warped_binary * 255):
        raise RuntimeError(f"Could not save image: {warped_path}")

    if not cv2.imwrite(lane_pixels_path, lane_pixels_image):
        raise RuntimeError(f"Could not save image: {lane_pixels_path}")

    if not cv2.imwrite(fitted_lane_path, fitted_lane_image):
        raise RuntimeError(f"Could not save image: {fitted_lane_path}")

    if not cv2.imwrite(final_lane_path, final_lane_image):
        raise RuntimeError(f"Could not save image: {final_lane_path}")

    print(f"Saved undistorted image: {undistorted_path}")
    print(f"Saved original vs undistorted image: {comparison_path}")
    print(f"Saved binary image: {binary_path}")
    print(f"Saved warped binary image: {warped_path}")
    print(f"Saved lane pixel image: {lane_pixels_path}")
    print(f"Saved fitted lane image: {fitted_lane_path}")
    print(f"Saved final lane image: {final_lane_path}")
    print(f"Left lane pixels found: {len(left_x)}")
    print(f"Right lane pixels found: {len(right_x)}")
    print(f"Left polynomial: {left_fit}")
    print(f"Right polynomial: {right_fit}")
    print(f"Lane curvature: {curvature:.2f} m")
    print(f"Vehicle offset: {vehicle_offset:.2f} m")

    print("Camera matrix:")
    print(camera_matrix)

    print("Distortion coefficients:")
    print(distortion_coeffs)

    process_video(
        os.path.join(test_videos_dir, "challenge01.mp4"),
        os.path.join(rnd_tests_output_dir, "challenge01_output.mp4"),
        camera_matrix,
        distortion_coeffs
    )
    process_video(
        os.path.join(test_videos_dir, "challenge02.mp4"),
        os.path.join(rnd_tests_output_dir, "challenge02_output.mp4"),
        camera_matrix,
        distortion_coeffs
    )
    process_video(
        os.path.join(test_videos_dir, "challenge03.mp4"),
        os.path.join(rnd_tests_output_dir, "challenge03_output.mp4"),
        camera_matrix,
        distortion_coeffs
    )
    process_video(
        os.path.join(test_videos_dir, "project_video03.mp4"),
        os.path.join(rnd_tests_output_dir, "project_video03_output.mp4"),
        camera_matrix,
        distortion_coeffs
    )
    process_video(
        os.path.join(test_videos_dir, "project_video02.mp4"),
        os.path.join(rnd_tests_output_dir, "project_video02_output.mp4"),
        camera_matrix,
        distortion_coeffs
    )
    process_video(
        os.path.join(test_videos_dir, "project_video01.mp4"),
        os.path.join(rnd_tests_output_dir, "project_video01_output.mp4"),
        camera_matrix,
        distortion_coeffs
    )

if __name__ == "__main__":
    main()
