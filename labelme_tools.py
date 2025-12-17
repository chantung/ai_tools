import json
import random
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ImageAugmentor:
    """labelme标注图像数据增强器，支持旋转、翻转、缩放、平移和颜色域更改，同时更新标注坐标"""
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str):
        """初始化增强器"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def get_image_json_pairs(self) -> List[Tuple[Path, Path]]:
        """获取图片和对应JSON文件的配对列表"""
        pairs = []
        
        # 遍历图片目录
        for img_path in self.input_dir.iterdir():
            if img_path.suffix.lower() in self.image_extensions:
                # 查找对应的JSON文件（假设同名）
                json_path = self.input_dir / f"{img_path.stem}.json"
                if json_path.exists():
                    pairs.append((img_path, json_path))
                else:
                    print(f"警告: 未找到 {img_path.name} 对应的JSON文件")
        
        print(f"找到 {len(pairs)} 对图像-JSON文件")
        return pairs
    
    def load_json_annotations(self, json_path: Path) -> Dict:
        """加载JSON标注文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_json_annotations(self, data: Dict, output_path: Path):
        """保存JSON标注文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def rotate_image_and_points(self, 
                               image: np.ndarray,
                               points: List[Tuple[float, float]],
                               angle: float) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        旋转图像和点坐标
        
        Args:
            image: 原始图像
            points: 点坐标列表 [(x1, y1), (x2, y2), ...]
            angle: 旋转角度（度），正数逆时针
        
        Returns:
            rotated_image: 旋转后的图像
            rotated_points: 旋转后的点坐标
        """
        h, w = image.shape[:2]
        # 使用浮点中心，避免整数舍入导致的微小偏移
        center = (w / 2.0, h / 2.0)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新图像边界
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        
        # 调整旋转矩阵的平移部分
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # 旋转图像
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114)  # 灰色填充
        )
        
        # 旋转点坐标
        rotated_points = []
        for point in points:
            if len(point) != 2:
                rotated_points.append(point)
                continue
                
            x, y = point
            # 转换为齐次坐标
            point_homogeneous = np.array([x, y, 1])
            rotated_point = rotation_matrix @ point_homogeneous
            
            # 确保坐标在图像范围内
            rx = max(0, min(rotated_point[0], new_w - 1))
            ry = max(0, min(rotated_point[1], new_h - 1))
            rotated_points.append((float(rx), float(ry)))
        
        return rotated_image, rotated_points

    def _apply_affine_to_point(self, M: np.ndarray, x: float, y: float, new_w: int, new_h: int) -> Tuple[float, float]:
        """
        将仿射矩阵应用到单个点，并进行越界裁剪
        """
        p = M @ np.array([x, y, 1.0])
        rx = max(0.0, min(float(p[0]), float(new_w - 1)))
        ry = max(0.0, min(float(p[1]), float(new_h - 1)))
        return rx, ry

    def rotate_image_and_json(self,
                              image: np.ndarray,
                              json_data: Dict,
                              angle: float) -> Tuple[np.ndarray, Dict]:
        """
        旋转图像并按 labelme 结构更新 JSON 标注（正确处理 rectangle）
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 计算旋转矩阵与新画布尺寸
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # 平移使图像位于新画布中心
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0

        # 旋转图像
        rotated_image = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114)
        )

        # 更新 JSON（保持 labelme 的 shapes 结构）
        updated_json = json_data.copy()
        updated_json['imageWidth'] = new_w
        updated_json['imageHeight'] = new_h
        # 保证增强信息字段存在
        if 'augmentation_info' not in updated_json:
            updated_json['augmentation_info'] = []

        if 'shapes' in updated_json:
            for shape in updated_json['shapes']:
                shape_type = shape.get('shape_type', '')
                pts = shape.get('points', [])

                if shape_type == 'rectangle' and len(pts) == 2:
                    # rectangle 是轴对齐矩形，旋转后无法保持旋转角度
                    # 这里将其转换为旋转后四点的包围盒（轴对齐），保证标注合法
                    (x1, y1), (x2, y2) = pts
                    rect_corners = [
                        (x1, y1),
                        (x1, y2),
                        (x2, y1),
                        (x2, y2),
                    ]
                    rotated_corners = [self._apply_affine_to_point(M, x, y, new_w, new_h) for x, y in rect_corners]
                    xs = [c[0] for c in rotated_corners]
                    ys = [c[1] for c in rotated_corners]
                    min_x, max_x = float(min(xs)), float(max(xs))
                    min_y, max_y = float(min(ys)), float(max(ys))
                    shape['points'] = [[min_x, min_y], [max_x, max_y]]
                else:
                    # point / polygon 等直接逐点仿射变换
                    for i in range(len(pts)):
                        x, y = float(pts[i][0]), float(pts[i][1])
                        rx, ry = self._apply_affine_to_point(M, x, y, new_w, new_h)
                        pts[i] = [rx, ry]

        # 若为 COCO 或简单 points 格式，尽量保持兼容
        elif 'annotations' in updated_json:
            for ann in updated_json['annotations']:
                if 'keypoints' in ann:
                    kps = ann['keypoints']
                    for i in range(0, len(kps), 3):
                        x, y = float(kps[i]), float(kps[i+1])
                        rx, ry = self._apply_affine_to_point(M, x, y, new_w, new_h)
                        kps[i], kps[i+1] = rx, ry
        elif 'points' in updated_json:
            pts = updated_json['points']
            new_pts = []
            for p in pts:
                x, y = float(p[0]), float(p[1])
                rx, ry = self._apply_affine_to_point(M, x, y, new_w, new_h)
                new_pts.append([rx, ry])
            updated_json['points'] = new_pts

        return rotated_image, updated_json
    
    def scale_image_and_json(self,
                             image: np.ndarray,
                             json_data: Dict,
                             scale_x: float,
                             scale_y: float) -> Tuple[np.ndarray, Dict]:
        """
        缩放图像并同步更新 JSON 标注坐标
        
        Args:
            image: 原始图像
            json_data: 标注 JSON（支持 labelme 的 shapes、COCO 的 keypoints 或简单 points）
            scale_x: 水平方向缩放比例
            scale_y: 垂直方向缩放比例
        
        Returns:
            scaled_image: 缩放后的图像
            updated_json: 按比例缩放并裁剪后的 JSON 标注
        """
        h, w = image.shape[:2]
        new_w = max(1, int(w * scale_x))
        new_h = max(1, int(h * scale_y))
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        updated_json = json_data.copy()
        updated_json['imageWidth'] = new_w
        updated_json['imageHeight'] = new_h
        if 'augmentation_info' not in updated_json:
            updated_json['augmentation_info'] = []
        if 'shapes' in updated_json:
            for shape in updated_json['shapes']:
                pts = shape.get('points', [])
                for i in range(len(pts)):
                    x, y = float(pts[i][0]), float(pts[i][1])
                    rx = max(0.0, min(x * scale_x, new_w - 1))
                    ry = max(0.0, min(y * scale_y, new_h - 1))
                    pts[i] = [rx, ry]
        elif 'annotations' in updated_json:
            for ann in updated_json['annotations']:
                if 'keypoints' in ann:
                    kps = ann['keypoints']
                    for i in range(0, len(kps), 3):
                        kps[i] = float(kps[i]) * scale_x
                        kps[i+1] = float(kps[i+1]) * scale_y
        elif 'points' in updated_json:
            pts = updated_json['points']
            new_pts = []
            for p in pts:
                x, y = float(p[0]), float(p[1])
                new_pts.append([max(0.0, min(x * scale_x, new_w - 1)),
                                max(0.0, min(y * scale_y, new_h - 1))])
            updated_json['points'] = new_pts
        return scaled_image, updated_json
    
    def translate_image_and_json(self,
                                 image: np.ndarray,
                                 json_data: Dict,
                                 tx: float,
                                 ty: float) -> Tuple[np.ndarray, Dict]:
        """
        平移图像并同步更新 JSON 标注坐标
        
        Args:
            image: 原始图像
            json_data: 标注 JSON（支持 labelme 的 shapes、COCO 的 keypoints 或简单 points）
            tx: 水平方向平移像素（正右负左）
            ty: 垂直方向平移像素（正下负上）
        
        Returns:
            translated_image: 平移后的图像
            updated_json: 平移并裁剪后的 JSON 标注
        """
        h, w = image.shape[:2]
        M = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty]], dtype=np.float32)
        translated_image = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114)
        )
        updated_json = json_data.copy()
        updated_json['imageWidth'] = w
        updated_json['imageHeight'] = h
        if 'augmentation_info' not in updated_json:
            updated_json['augmentation_info'] = []
        if 'shapes' in updated_json:
            for shape in updated_json['shapes']:
                pts = shape.get('points', [])
                for i in range(len(pts)):
                    x, y = float(pts[i][0]), float(pts[i][1])
                    rx = max(0.0, min(x + tx, w - 1))
                    ry = max(0.0, min(y + ty, h - 1))
                    pts[i] = [rx, ry]
        elif 'annotations' in updated_json:
            for ann in updated_json['annotations']:
                if 'keypoints' in ann:
                    kps = ann['keypoints']
                    for i in range(0, len(kps), 3):
                        kps[i] = max(0.0, min(float(kps[i]) + tx, w - 1))
                        kps[i+1] = max(0.0, min(float(kps[i+1]) + ty, h - 1))
        elif 'points' in updated_json:
            pts = updated_json['points']
            new_pts = []
            for p in pts:
                x, y = float(p[0]), float(p[1])
                new_pts.append([max(0.0, min(x + tx, w - 1)),
                                max(0.0, min(y + ty, h - 1))])
            updated_json['points'] = new_pts
        return translated_image, updated_json
    
    def adjust_brightness_contrast(self,
                                   image: np.ndarray,
                                   alpha: float,
                                   beta: float) -> np.ndarray:
        """
        调整图像亮度与对比度（不修改标注坐标）
        
        Args:
            image: 原始图像
            alpha: 对比度因子（>1 增强，<1 减弱）
            beta: 亮度偏移（正变亮，负变暗），单位像素值
        
        Returns:
            adjusted_image: 亮度/对比度调整后的图像
        """
        img = image.astype(np.float32)
        img = img * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    
    def flip_image_and_points(self,
                             image: np.ndarray,
                             points: List[Tuple[float, float]],
                             flip_code: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        翻转图像和点坐标
        
        Args:
            image: 原始图像
            points: 点坐标列表 [(x1, y1), (x2, y2), ...]
            flip_code: 0-垂直翻转, 1-水平翻转, -1-水平和垂直翻转
        
        Returns:
            flipped_image: 翻转后的图像
            flipped_points: 翻转后的点坐标
        """
        h, w = image.shape[:2]
        
        # 翻转图像
        flipped_image = cv2.flip(image, flip_code)
        
        # 翻转点坐标
        flipped_points = []
        for point in points:
            if len(point) != 2:
                flipped_points.append(point)
                continue
                
            x, y = point
            if flip_code == 0:  # 垂直翻转
                flipped_x = x
                flipped_y = h - y - 1
            elif flip_code == 1:  # 水平翻转
                flipped_x = w - x - 1
                flipped_y = y
            else:  # 水平和垂直翻转
                flipped_x = w - x - 1
                flipped_y = h - y - 1
            
            # 确保坐标在图像范围内
            flipped_x = max(0, min(flipped_x, w - 1))
            flipped_y = max(0, min(flipped_y, h - 1))
            flipped_points.append((float(flipped_x), float(flipped_y)))
        
        return flipped_image, flipped_points
    
    def extract_points_from_json(self, json_data: Dict) -> Tuple[List[Tuple[float, float]], Dict]:
        """
        从JSON数据中提取点坐标和元数据
        
        Args:
            json_data: JSON标注数据
            
        Returns:
            points: 点坐标列表
            metadata: 其他元数据
        """
        points = []
        metadata = {}
        
        # 根据常见的标注格式处理
        # 格式1: COCO格式
        if 'annotations' in json_data:
            annotations = json_data['annotations']
            for ann in annotations:
                if 'keypoints' in ann:
                    keypoints = ann['keypoints']
                    # 假设格式为 [x1, y1, v1, x2, y2, v2, ...]
                    for i in range(0, len(keypoints), 3):
                        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                        if v > 0:  # 只处理可见点
                            points.append((float(x), float(y)))
        
        # 格式2: 自定义格式，包含points字段
        elif 'shapes' in json_data:
            shapes = json_data['shapes']
            for shape in shapes:
                if shape['shape_type'] == 'point':
                    for point in shape['points']:
                        x, y = point
                        points.append((float(x), float(y)))
                elif shape['shape_type'] in ['polygon', 'rectangle']:
                    for point in shape['points']:
                        x, y = point
                        points.append((float(x), float(y)))
        
        # 格式3: 简单的点列表
        elif 'points' in json_data:
            points = [(float(p[0]), float(p[1])) for p in json_data['points']]
        
        # 保存其他元数据
        for key, value in json_data.items():
            if key not in ['points', 'annotations', 'shapes']:
                metadata[key] = value
        
        return points, metadata
    
    def update_json_with_new_points(self, 
                                   original_json: Dict,
                                   new_points: List[Tuple[float, float]],
                                   image_size: Tuple[int, int]) -> Dict:
        """
        用新的点坐标更新JSON数据
        
        Args:
            original_json: 原始JSON数据
            new_points: 新的点坐标列表
            image_size: 新图像的尺寸 (width, height)
            
        Returns:
            updated_json: 更新后的JSON数据
        """
        updated_json = original_json.copy()
        
        # 更新图像尺寸信息
        updated_json['imageWidth'] = image_size[0]
        updated_json['imageHeight'] = image_size[1]
        
        # 根据原始格式更新点坐标
        # 格式1: COCO格式
        if 'annotations' in updated_json:
            point_idx = 0
            for ann in updated_json['annotations']:
                if 'keypoints' in ann:
                    keypoints = ann['keypoints']
                    for i in range(0, len(keypoints), 3):
                        if point_idx < len(new_points):
                            x, y = new_points[point_idx]
                            keypoints[i] = x
                            keypoints[i+1] = y
                            point_idx += 1
        
        # 格式2: 自定义格式
        elif 'shapes' in updated_json:
            point_idx = 0
            for shape in updated_json['shapes']:
                if shape['shape_type'] == 'point':
                    for i in range(len(shape['points'])):
                        if point_idx < len(new_points):
                            shape['points'][i] = list(new_points[point_idx])
                            point_idx += 1
                elif shape['shape_type'] in ['polygon', 'rectangle']:
                    for i in range(len(shape['points'])):
                        if point_idx < len(new_points):
                            shape['points'][i] = list(new_points[point_idx])
                            point_idx += 1
        
        # 格式3: 简单的点列表
        elif 'points' in updated_json:
            updated_json['points'] = [list(p) for p in new_points]
        
        # 添加增强信息
        if 'augmentation_info' not in updated_json:
            updated_json['augmentation_info'] = []
        
        return updated_json
    
    def augment_pair(self, 
                    img_path: Path, 
                    json_path: Path,
                    augment_type: Optional[str] = None) -> List[Tuple[Path, Path]]:
        """
        对一对图像和JSON进行增强
        
        Args:
            img_path: 图像路径
            json_path: JSON路径
            augment_type: 增强类型 ('rotate', 'flip', 'both')，None表示随机选择
            
        Returns:
            List of (new_image_path, new_json_path) pairs
        """
        # 加载图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"错误: 无法加载图像 {img_path}")
            return []
        
        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载JSON标注
        json_data = self.load_json_annotations(json_path)
        
        # 提取点坐标
        original_points, metadata = self.extract_points_from_json(json_data)
        
        # 如果没有点坐标，只处理图像
        if not original_points:
            print(f"警告: {json_path} 中没有找到点坐标")
            return []
        
        results = []
        original_h, original_w = image.shape[:2]
        
        # 确定增强类型
        if augment_type is None:
            augment_type = random.choice(['rotate', 'flip', 'both'])
        
        # 旋转增强（按 labelme 结构直接更新 JSON）
        if augment_type in ['rotate', 'both']:
            angle = random.uniform(-15, 15)

            rotated_image, rotated_json = self.rotate_image_and_json(image, json_data, angle)

            rotated_h, rotated_w = rotated_image.shape[:2]
            # 保证增强信息字段存在
            if 'augmentation_info' not in rotated_json:
                rotated_json['augmentation_info'] = []
            rotated_json['augmentation_info'].append({
                'type': 'rotation',
                'angle': float(angle),
                'original_size': [original_w, original_h],
                'new_size': [rotated_w, rotated_h]
            })

            suffix = f"_rot{angle:.1f}"
            new_img_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            new_json_name = f"{json_path.stem}{suffix}.json"

            new_img_path = self.output_dir / new_img_name
            new_json_path = self.output_dir / new_json_name

            rotated_image_bgr = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), rotated_image_bgr)

            rotated_json['imagePath'] = new_img_name
            self.save_json_annotations(rotated_json, new_json_path)

            results.append((new_img_path, new_json_path))
            print(f"创建旋转增强: {new_img_name}, 角度: {angle:.1f}度")
        
        # 缩放增强
        if augment_type == 'scale':
            scale_x = random.uniform(0.8, 1.2)
            scale_y = random.uniform(0.8, 1.2)
            scaled_image, scaled_json = self.scale_image_and_json(image, json_data, scale_x, scale_y)
            sh, sw = scaled_image.shape[:2]
            if 'augmentation_info' not in scaled_json:
                scaled_json['augmentation_info'] = []
            scaled_json['augmentation_info'].append({
                'type': 'scale',
                'scale_x': float(scale_x),
                'scale_y': float(scale_y),
                'original_size': [original_w, original_h],
                'new_size': [sw, sh]
            })
            suffix = f"_scale{scale_x:.2f}x{scale_y:.2f}"
            new_img_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            new_json_name = f"{json_path.stem}{suffix}.json"
            new_img_path = self.output_dir / new_img_name
            new_json_path = self.output_dir / new_json_name
            scaled_image_bgr = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), scaled_image_bgr)
            scaled_json['imagePath'] = new_img_name
            self.save_json_annotations(scaled_json, new_json_path)
            results.append((new_img_path, new_json_path))
            print(f"创建缩放增强: {new_img_name}, 比例: ({scale_x:.2f}, {scale_y:.2f})")
        
        # 平移增强
        if augment_type == 'translate':
            max_tx = int(0.1 * original_w)
            max_ty = int(0.1 * original_h)
            tx = random.randint(-max_tx, max_tx)
            ty = random.randint(-max_ty, max_ty)
            translated_image, translated_json = self.translate_image_and_json(image, json_data, tx, ty)
            th, tw = translated_image.shape[:2]
            if 'augmentation_info' not in translated_json:
                translated_json['augmentation_info'] = []
            translated_json['augmentation_info'].append({
                'type': 'translate',
                'tx': int(tx),
                'ty': int(ty),
                'original_size': [original_w, original_h],
                'new_size': [tw, th]
            })
            suffix = f"_trans{tx}_{ty}"
            new_img_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            new_json_name = f"{json_path.stem}{suffix}.json"
            new_img_path = self.output_dir / new_img_name
            new_json_path = self.output_dir / new_json_name
            translated_image_bgr = cv2.cvtColor(translated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), translated_image_bgr)
            translated_json['imagePath'] = new_img_name
            self.save_json_annotations(translated_json, new_json_path)
            results.append((new_img_path, new_json_path))
            print(f"创建平移增强: {new_img_name}, 位移: ({tx}, {ty})")
        
        # 亮度/对比度增强（不改变 JSON 坐标）
        if augment_type == 'color':
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-20, 20)
            color_image = self.adjust_brightness_contrast(image, alpha, beta)
            color_json = json_data.copy()
            if 'augmentation_info' not in color_json:
                color_json['augmentation_info'] = []
            color_json['augmentation_info'].append({
                'type': 'brightness_contrast',
                'alpha': float(alpha),
                'beta': float(beta),
                'original_size': [original_w, original_h],
                'new_size': [original_w, original_h]
            })
            suffix = f"_bc{alpha:.2f}_{int(beta)}"
            new_img_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            new_json_name = f"{json_path.stem}{suffix}.json"
            new_img_path = self.output_dir / new_img_name
            new_json_path = self.output_dir / new_json_name
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), color_image_bgr)
            color_json['imagePath'] = new_img_name
            self.save_json_annotations(color_json, new_json_path)
            results.append((new_img_path, new_json_path))
            print(f"创建亮度/对比度增强: {new_img_name}, alpha: {alpha:.2f}, beta: {beta:.1f}")
        
        # 翻转增强
        if augment_type in ['flip', 'both']:
            # 随机选择翻转类型
            flip_types = [0, 1]  # 0:垂直翻转, 1:水平翻转
            flip_code = random.choice(flip_types)
            
            # 翻转图像和点
            flipped_image, flipped_points = self.flip_image_and_points(
                image, original_points, flip_code
            )
            
            # 更新JSON数据
            flipped_h, flipped_w = flipped_image.shape[:2]
            flipped_json = self.update_json_with_new_points(
                json_data, flipped_points, (flipped_w, flipped_h)
            )
            
            # 添加翻转信息
            flip_name = 'vertical' if flip_code == 0 else 'horizontal'
            flipped_json['augmentation_info'].append({
                'type': 'flip',
                'flip_type': flip_name,
                'original_size': [original_w, original_h],
                'new_size': [flipped_w, flipped_h]
            })
            
            # 生成文件名
            suffix = f"_flip{flip_name[0]}"
            new_img_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            new_json_name = f"{json_path.stem}{suffix}.json"
            
            new_img_path = self.output_dir / new_img_name
            new_json_path = self.output_dir / new_json_name
            
            # 保存图像
            flipped_image_bgr = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), flipped_image_bgr)

            flipped_json['imagePath'] = new_img_name
            
            # 保存JSON
            self.save_json_annotations(flipped_json, new_json_path)
            
            results.append((new_img_path, new_json_path))
            print(f"创建翻转增强: {new_img_name}, 类型: {flip_name}")
        
        return results
    
    def run_augmentation(self, 
                        num_augmentations: int = 1,
                        random_seed: int = 42):
        """
        运行数据增强
        
        Args:
            num_augmentations: 每个原始样本生成多少增强样本
            random_seed: 随机种子
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 获取所有图像-JSON对
        pairs = self.get_image_json_pairs()
        
        if not pairs:
            print("错误: 未找到任何图像-JSON对")
            return
        
        total_generated = 0
        
        # 对每对进行增强
        for img_path, json_path in pairs:
            print(f"\n处理: {img_path.name}")
            
            for i in range(num_augmentations):
                # 随机选择增强类型
                augment_type = random.choice(['rotate', 'flip', 'scale', 'translate', 'color', 'both'])
                
                # 进行增强
                results = self.augment_pair(img_path, json_path, augment_type)
                total_generated += len(results)
        
        print(f"\n完成! 总共生成 {total_generated} 个增强样本")
        print(f"原始图像保存在: {self.input_dir}")
        print(f"增强图像保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="LabelMe Image Augmentation Tool")
    parser.add_argument("--src_dir", type=str, required=True, help="Input directory containing images and json files")
    parser.add_argument("--dist_dir", type=str, required=True, help="Output directory for augmented dataset")
    parser.add_argument("--num_aug", type=int, default=3, help="Number of augmented samples per original image")
    
    args = parser.parse_args()
    
    # 创建增强器
    augmentor = ImageAugmentor(
        input_dir=args.src_dir,
        output_dir=args.dist_dir,
    )
    
    # 运行增强
    augmentor.run_augmentation(num_augmentations=args.num_aug, random_seed=42)

# 使用示例
if __name__ == "__main__":
    main()
