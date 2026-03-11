import torch


def compute_metrics(pred_age, age, pred_gender, gender, pred_race, race):

    pred_age = pred_age.squeeze()

    mae = torch.mean(torch.abs(pred_age - age)).item()

    gender_pred = pred_gender.argmax(dim=1)
    gender_acc = (gender_pred == gender).float().mean().item()

    race_pred = pred_race.argmax(dim=1)
    race_acc = (race_pred == race).float().mean().item()

    return mae, gender_acc, race_acc

