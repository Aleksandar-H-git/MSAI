import os
import cv2


camera_cal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_cal")
image_list= []
rnd_tests_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_tests")
def main():
    print(camera_cal_dir)
    for file_name in os.listdir(camera_cal_dir):
        if file_name.lower().endswith((".jpg",".jpeg")):
                img_path = os.path.join(camera_cal_dir,file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_list.append(img)
    cnt = 1
    for img in image_list:        
        cv2.imwrite(os.path.join(rnd_tests_output_dir, "img" + str(cnt) + ".jpg"), img)
        cnt += 1
    return


if __name__ == "__main__":
    main()