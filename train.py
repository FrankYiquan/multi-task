import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import UTKFaceDataset
from model.model import MultiTaskEfficientNet
from utils.eval import compute_metrics


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = UTKFaceDataset("data/train", train_transform)
    val_dataset = UTKFaceDataset("data/val", val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    model = MultiTaskEfficientNet().to(device)

    age_loss = nn.L1Loss()
    # gender_loss = nn.CrossEntropyLoss()
    # race_loss = nn.CrossEntropyLoss()
    gender_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    race_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    os.makedirs("output", exist_ok=True)

    best_score = -float("inf")

    for epoch in range(args.epochs):

        model.train()

        for images, age, gender, race in tqdm(train_loader):

            images = images.to(device)
            age = age.float().to(device)
            gender = gender.to(device)
            race = race.to(device)

            optimizer.zero_grad()

            pred_age, pred_gender, pred_race = model(images)
            pred_age = pred_age.squeeze()

            loss_age = age_loss(pred_age, age)
            loss_gender = gender_loss(pred_gender, gender)
            loss_race = race_loss(pred_race, race)

            loss = 0.5 * loss_age + loss_gender + 1.5 * loss_race

            loss.backward()
            optimizer.step()

        print(f"\nEpoch {epoch+1} finished")

        # Evaluate training set using final model
        print("Training Results")
        train_mae, train_gender, train_race = evaluate(model, train_loader, device)
        print("Age MAE:", train_mae)
        print("Gender Acc:", train_gender)
        print("Race Acc:", train_race)

        # Validation
        print()
        print("Validation Results")
        val_mae, val_gender, val_race = evaluate(model, val_loader, device)
        print("Age MAE:", val_mae)  
        print("Gender Acc:", val_gender)
        print("Race Acc:", val_race)

        score = val_gender + val_race - val_mae

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "output/model.pt")


def evaluate(model, loader, device):

    model.eval()

    mae_total = 0
    gender_acc_total = 0
    race_acc_total = 0
    count = 0

    with torch.no_grad():

        for images, age, gender, race in loader:

            images = images.to(device)
            age = age.float().to(device)
            gender = gender.to(device)
            race = race.to(device)

            pred_age, pred_gender, pred_race = model(images)
            pred_age = pred_age.squeeze()

            mae, gender_acc, race_acc = compute_metrics(
                pred_age, age,
                pred_gender, gender,
                pred_race, race
            )

            batch_size = images.size(0)

            mae_total += mae * batch_size
            gender_acc_total += gender_acc * batch_size
            race_acc_total += race_acc * batch_size
            count += batch_size

    val_mae = mae_total / count
    val_gender = gender_acc_total / count
    val_race = race_acc_total / count

    return val_mae, val_gender, val_race