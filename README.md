### AI-Tools
记录一些ML和AI相关的小工具

#### 训练类
1. ##### labelme_tools.py

labelme标注格式的图片的增强工具

使用方法

```python
python labelme_tools.py --src_dir /path/to/your/labelme/image --dist_dir /path/to/output --num_aug 3
```

* `src_dir`: 已经使用labelme标注过的图片路径
* `dist_dir`：增样后的输出路径
* `num_aug`：每个图片创建几个增强样本

2. ##### image_augmentor.py

基于albumentations的一些图片变换
