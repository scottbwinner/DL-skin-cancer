import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from sklearn.metrics import f1_score
import copy

def train(
    model,
    model_name: str,
    train_loader,
    val_loader,
    loss_func: any,
    optimizer: any,
    exp_dir: str = "models",
    log_dir: str = "logs",
    num_epoch: int = 50,
    #batch_size: int = 128,
    seed: int = 2026,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    full_model_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir = Path(log_dir) / full_model_name
    exp_dir = Path(exp_dir)
    logger = tb.SummaryWriter(log_dir)

    model = model.to(device)
    loss_func = loss_func.to(device)
    max_val_f1 = -1

    best_state_dict = {}

    global_step = 0
    metrics = {"train_acc": [], "val_acc": [], "train_macro_f1": [], "val_macro_f1": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        all_train_preds = []
        all_train_labels = []
        all_val_preds = []
        all_val_labels = []

        model.train() # set model to training mode (parameter training = True)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Compute gradient and update weights
            optimizer.zero_grad()  # zero the parameter gradients
            logits = model(images)  # forward pass calculates logit predictions for images
            loss = loss_func(logits, labels)  # compute loss
            logger.add_scalar('train_loss', loss, global_step)
            preds = logits.argmax(dim=1)
            train_accuracy = (preds == labels).float().mean() # add 1 if correct, 0 if incorrect

            metrics["train_acc"].append(train_accuracy)

            all_train_preds.append(preds.cpu().numpy())
            all_train_labels.append(labels.cpu().numpy())

            loss.backward()  # backward pass computes gradients for each parameter
            optimizer.step()  # update weights using the computed gradients

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval() # set model to evaluation mode (parameter training = False)
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)  # forward pass calculates logit predictions for img
                preds = logits.argmax(dim=1)
                val_accuracy = (preds == labels).float().mean()
                metrics["val_acc"].append(val_accuracy)
                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())
                

        epoch_train_macro_f1 = f1_score(
            y_true=np.concatenate(all_train_labels),
            y_pred=np.concatenate(all_train_preds),
            average='macro',
            zero_division=0,
        )
        epoch_val_macro_f1 = f1_score(
            y_true=np.concatenate(all_val_labels),
            y_pred=np.concatenate(all_val_preds),
            average='macro',
            zero_division=0,
        )

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()


        if epoch_val_macro_f1 > max_val_f1:
            max_val_f1 = epoch_val_macro_f1
            best_state_dict = copy.deepcopy(model.state_dict())

        logger.add_scalar("train_acc", epoch_train_acc, global_step)
        logger.add_scalar("val_acc", epoch_val_acc, global_step)
        logger.add_scalar("train_f1", epoch_train_macro_f1, global_step)
        logger.add_scalar("val_f1", epoch_val_macro_f1, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"train_f1={epoch_train_macro_f1:.4f} "
                f"val_f1={epoch_val_macro_f1:.4f} "
            )



    # save a copy of model weights in the log directory
    torch.save(best_state_dict, exp_dir / f"{full_model_name}.th")
    print(f"Model saved to {log_dir / f'{full_model_name}.th'}")

    # return trained model
    model.load_state_dict(best_state_dict)
    return model