import os
from PIL import Image

IMG_SIZE = (384, 384)

INPUT_TRAIN = r"c:/Users/jfbaa/Downloads/dataforreid/train/train"
INPUT_TEST = r"c:/Users/jfbaa/Downloads/dataforreid/test/test"

OUTPUT_TRAIN = r"c:/Users/jfbaa/OneDrive/Documents/re-idversion2/train_384x384"
OUTPUT_TEST = r"c:/Users/jfbaa/OneDrive/Documents/re-idversion2/test_384x384"


def crop_alphachannel(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        bbox = img.getbbox()
        if bbox:
            return img.crop(bbox)
    return img


def preprocess_and_save(input_dir, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    total = len(files)
    print(f"Processing {total} {label} images...")

    for i, fname in enumerate(files, 1):
        img_path = os.path.join(input_dir, fname)
        try:
            image = Image.open(img_path).convert("RGBA")
            image = crop_alphachannel(image)
            image = image.convert("RGB")
            image = image.resize(IMG_SIZE, Image.LANCZOS)
            out_path = os.path.join(output_dir, fname)
            image.save(out_path)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")

        if i % 200 == 0 or i == total:
            print(f"  {i}/{total} done")

    print(f"Saved to {output_dir}\n")


if __name__ == "__main__":
    preprocess_and_save(INPUT_TRAIN, OUTPUT_TRAIN, "train")
    preprocess_and_save(INPUT_TEST, OUTPUT_TEST, "test")
    print("All done!")
