import cv2
import os
import re
from tqdm import tqdm

# 设置输入文件夹路径和输出视频文件路径
source_folder = '/home/guest/Documents/Nymeria/20231222_s1_kenneth_fischer_act7_56uvqd/recording_head/imgs_1049_1990'
input_folder = '/home/guest/github/hold/out_demo'  # 替换成你实际的文件夹路径
output_video = '/home/guest/github/hold/output_video.mp4'  # 输出的视频文件

# 获取文件夹内的所有符合 xxxx_all.jpg 格式的图片
ori_image_files = [f for f in os.listdir(source_folder) if re.match(r'\d+\.\d+\.jpg', f)]
image_files = [f for f in os.listdir(input_folder) if re.match(r'\d+\.\d+_all\.jpg', f)]

# 按照文件名中的时间戳（浮点数）进行排序
ori_image_files.sort(key=lambda x: float(re.match(r'(\d+\.\d+)\.jpg', x).group(1)))

# 读取第一张图片来获取视频的尺寸
first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_image.shape

# 设置视频写入对象，使用mp4格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 你可以使用 'XVID' 或其他格式
video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

# 遍历图片并将它们写入视频
for idx, ori_img_file in enumerate(tqdm(ori_image_files, desc="Processing Images", ncols=100)):
    if f'{ori_img_file[:-4]}_all.jpg' in image_files:
        image_path = os.path.join(input_folder, f'{ori_img_file[:-4]}_all.jpg')
    else:
        image_path = os.path.join(source_folder, ori_img_file)
    img = cv2.imread(image_path)
    
    # 提取时间戳并将其显示在图像的左上角
    timestamp = float(re.match(r'(\d+\.\d+)\.jpg', ori_img_file).group(1))
    cv2.putText(img, f'{idx}: {timestamp:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 写入视频
    video_writer.write(img)

# 释放视频写入对象
video_writer.release()

print(f"Video saved to {output_video}")
