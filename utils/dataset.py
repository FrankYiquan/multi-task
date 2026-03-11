import os
from PIL import Image
from torch.utils.data import Dataset


class UTKFaceDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.transform = transform

    def parse_labels(self, filename):

        name = filename.split(".")[0]

        # remove prefix  (part1__)
        label_part = name.split("__")[-1]

        age, gender, race = label_part.split("_")[:3]

        return int(age), int(gender), int(race)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        path = os.path.join(self.root_dir, file)

        image = Image.open(path).convert("RGB")

        age, gender, race = self.parse_labels(file)

        if self.transform:
            image = self.transform(image)

        return image, age, gender, race