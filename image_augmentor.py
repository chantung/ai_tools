import albumentations as A
import cv2
from pathlib import Path
from tqdm import tqdm


def get_augmentation_pipline():
  return A.Compose([
    # 几何变换
    A.HorizontalFlip(p=0.5), # 水平翻转
    A.VerticalFlip(p=0.5), # 垂直翻转
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.7),
    # 像素变换
    A.OneOf([
      A.GaussianBlur(blur_limit=(3, 7), p=1), # 高斯模糊
      A.GaussNoise(std_range=(0.02, 0.1), p=1), # 高斯噪声
    ], p=0.5),
    A.RandomBrightnessContrast(p=0.5), # 随机亮度和对比度
    A.HueSaturationValue(p=0.3), # 色相、饱和度、明度调整
  ])

def process_images(input_dir, output_dir, num_aug=3):
  transform = get_augmentation_pipline()
  input_path = Path(input_dir)
  output_path = Path(output_dir)

  for img_path in tqdm(input_path.iterdir(), desc="Processing"):
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(num_aug):
      try:
        augmented = transform(image=image)['image']
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        file_name = f"{img_path.stem}_aug_{i+1}{img_path.suffix}"
        file_path = output_path / file_name
        cv2.imwrite(str(file_path), augmented)
      except Exception as e:
        print(f"处理图片 {img_path} 时出错: {e}")
  
  print(f"处理完成！所有增强后的图片已保存至: {output_dir}")


if __name__ == "__main__":
  INPUT_DIR = r""
  OUTPUT_DIR = r""
  process_images(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

