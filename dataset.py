import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import Config


def crop_alphachannel(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        bbox = img.getbbox()
        if bbox:
            return img.crop(bbox)
    return img


class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        try:
            image = Image.open(img_path).convert("RGBA")
            image = crop_alphachannel(image)
            image = image.convert("RGB")
        except Exception:
            image = Image.new("RGB", Config.IMG_SIZE)

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            return image, 0
        else:
            return image, row['filename']


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
