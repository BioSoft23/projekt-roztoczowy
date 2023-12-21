import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from collections import defaultdict
import pprint as pp
from torchvision import transforms


class YOLODataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = Path(directory)
        self.transform = transform
        # self.images = sorted(Path(f"{self.directory}/images").glob(".jpg"))
        # self.images = sorted((self.directory / "images").glob("*.jpg"))
        # self.annotations = sorted(self.directory.glob(".json"))

        # print(self.directory.as_posix())
        # print(list((self.directory / "images").glob("*.jpg")))
        with open(next(self.directory.glob("*.json"))) as f:
            self.annotations = json.load(f)

        new_anno = defaultdict(list)
        for anno in self.annotations['annotations']:
            image_id = anno['image_id']
            new_anno[image_id].append(anno)
        self.annotations['annotations'] = dict(new_anno)

        with open("anno_test.json", 'w') as f:
            json.dump(self.annotations, f, indent=4)

        self.images = sorted((self.directory / "images").glob("*.jpg"))
        self.images = [self.directory / img['file_name'] for img in self.annotations['images'] ]
        pp.pprint(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # img_path = os.path.join(self.directory, self.images[index])
        img_path = self.images[index]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        new_size = (400, 225)
        img = img.resize(new_size)
        img = transforms.ToTensor()(img)
        target = {}
        boxes = []
        labels = []
        areas = []
        crowds = []
        width_scale = new_size[0] / width
        print(width_scale)
        height_scale = new_size[1] / height
        print(height_scale)
        # print(self.annotations['annotations'].keys())
        for anno in self.annotations['annotations'][index]:
            scaled_boxes = [anno['bbox'][0] * width_scale, anno['bbox'][1] * height_scale, anno['bbox'][2] * width_scale, anno['bbox'][3] * height_scale]
            # boxes.append(anno['bbox'])
            boxes.append(scaled_boxes)
            labels.append(anno['category_id'])
            areas.append((anno['bbox'][2] * width_scale) * (anno['bbox'][3] * height_scale))
            crowds.append(anno['iscrowd'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # print(boxes)
        # there is only one class
        # labels = torch.ones((len(boxes),), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # print(labels)
        image_id = torch.tensor([index])
        # print(image_id)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # area = boxes[:, 3] * boxes[:, 2]
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # print(areas)
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        crowds = torch.as_tensor(crowds, dtype=torch.uint8)
        target["boxes"] = boxes
        target["class_labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = crowds
        if self.transform:
            img, target = self.transform(img, target)
        return img, target