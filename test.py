from config import get_args
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import UTKFaceDataset
from model.model import MultiTaskEfficientNet
from utils.eval import compute_metrics


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # consistsent with training/validation transforms 
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_dataset = UTKFaceDataset(
        root_dir=args.test_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Load model
    model = MultiTaskEfficientNet()
    model.load_state_dict(torch.load("output/model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    all_pred_age = []
    all_age = []

    all_pred_gender = []
    all_gender = []

    all_pred_race = []
    all_race = []

    with torch.no_grad():
        for images, age, gender, race in test_loader:

            images = images.to(device)
            age = age.to(device)
            gender = gender.to(device)
            race = race.to(device)

            age_pred, gender_pred, race_pred = model(images)

            all_pred_age.append(age_pred)
            all_age.append(age)

            all_pred_gender.append(gender_pred)
            all_gender.append(gender)

            all_pred_race.append(race_pred)
            all_race.append(race)

    # Concatenate all batches
    all_pred_age = torch.cat(all_pred_age)
    all_age = torch.cat(all_age)

    all_pred_gender = torch.cat(all_pred_gender)
    all_gender = torch.cat(all_gender)

    all_pred_race = torch.cat(all_pred_race)
    all_race = torch.cat(all_race)

    # Compute metrics
    age_mae, gender_acc, race_acc = compute_metrics(
        all_pred_age,
        all_age,
        all_pred_gender,
        all_gender,
        all_pred_race,
        all_race
    )

    print("\nTest Results")
    print("Age MAE:", age_mae)
    print("Gender Acc:", gender_acc)
    print("Race Acc:", race_acc)


if __name__ == "__main__":
    args = get_args()
    test(args)