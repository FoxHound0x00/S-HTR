import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from skimage.color import rgb2gray
from skimage.transform import rotate
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary

from utils import *
from dataset import SharadaDataset
from torch.utils.data import DataLoader
# from dataloader import SharadaDataLoader
from transforms import PadResize, Deskew, toRGB, ToTensor, Normalize_Cust, Grayscale, GaussianBlur, RandomErasing
from model import NewNet

os.makedirs("chk_pts/", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = (128,64)
dataset_dir = "/mnt/sda1/Sharada-stuff-June-2024/dataset_filter/dataset"


class_weights = calc_class_weights(tau=7, dataset_dir=dataset_dir)
print(class_weights)

null_annot, less_annot = get_null_annot(dataset_dir=dataset_dir)
print(null_annot, less_annot)
max_len = get_max_sequence_length(dataset_dir=dataset_dir)



dataset = SharadaDataset(files_dir=dataset_dir,
                        transform=Compose([
                            # Deslant(),
                            Grayscale(num_output_channels=1),
                            PadResize(output_size=(200,64)),
                            ToTensor(), # converted to Tensor
                            # Normalize_Cust(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                            Normalize_Cust(mean=[0],std=[1]),
                            GaussianBlur(kernel_size=5),
                            RandomErasing()
                        ]),
                        max_len=max_len,
                        null_annot=null_annot)
dl = DataLoader(dataset, batch_size[0], True)


crnn_model = NewNet(input_channels=1, hidden_size=64, 
                    num_classes=len(dataset.char_dict), 
                    max_seq_len=max_len).to(device=device)

optimizer = Adam(crnn_model.parameters(), lr=0.001)

# ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(list(class_weights.values())).to(device), reduction='mean')

# print(crnn_model)
summary(crnn_model, (100, 1, 64, 200))

def ce_loss_util(y, y_hat):
    y_hat = y_hat.reshape(-1)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[-1])
    loss = ce_loss(y, y_hat)
    return loss

# train_loader, val_loader = dl()


##### Visualization 
# data = next(iter(train_loader))
# img, label_, label_len = data
# print(f"Image Shape: {img.shape}, Label Shape: {label_.shape},  Label Lengths: {label_len}")
# print(f"Label: {label_[0]}")
# label_ = "".join([dataset.idx_to_char.get(c) for c in label_[0].cpu().numpy() if c!=-1])
# # print(f"Image : {img}, Label : {label_}")
# print(f"Maximum: {torch.max(img[0])}, Minimum: {torch.min(img[0])}")
# print(f"Label : {label_}")

# image = img[0].permute((1, 2, 0)).cpu()

# len_ = label_len[0]
# plt.imshow(image)

torch.autograd.set_detect_anomaly(True)

writer = SummaryWriter()
num_epochs = 50

# Training loop
for epoch in tqdm(range(num_epochs), desc='Training'):
    crnn_model.train()
    total_loss = 0.0
    tot_correct_tr = 0
    tot_samples_tr = 0
    i = 0
    # for images, targets, target_lengths, reshaped_targets in dl:
    for idx, sample in enumerate(dl):
        # print(sample)
        images, targets = sample["image"], sample["label"]
        # print("Batch No.",i)
        images = images.to(device)
        targets = targets.to(device) # Targets is [Batch, LongestSeqLen] (N,S) -> Targets should not be blanks
        # print(images.shape, targets.shape)
        
        # print(f"images: {images.shape}, Targets: {targets.shape}, lengths: {target_lengths.shape} ")
        optimizer.zero_grad()

        logits = crnn_model(images) # Outputs should be [TimeStep, Batch, NumClass]
        # (T,N,C) or (T,C) where C = number of characters in alphabet including blank, T = input length, and N = batch size.

        _, predicted_labels = torch.max(logits, 2)
        preds = ["".join([dataset.char_list[c] for c in row if c != 0]) for row in predicted_labels.cpu().numpy()]
        target_labels = ["".join([dataset.char_list[c] for c in row if c != 0]) for row in targets.cpu().numpy()]
        if epoch % 10 == 0 and idx == 0:
            print(f"Predicted: {preds[0]}, Target: {target_labels[0]}")
        correct = sum(pred == target for pred, target in zip(preds, target_labels))
        tot_correct_tr += correct
        tot_samples_tr += targets.size(0)


        # logits = torch.nn.functional.log_softmax(logits, dim=2)
        logit_lengths = torch.LongTensor([logits.size(0)] * logits.size(1))
        # [BatchSize] (N) Each must be <= T

        # logits = logits.transpose(0, 1)

        # print(f"LOGIT SHAPE {logits.shape} , TARGETS SHAPE {targets.shape}")

        # print(f"LOGIT SHAPE {logits.shape} , Reshaped TARGETS SHAPE {targets.shape}")
        # print(f" Logit Lengths : {logit_lengths.shape}  Target : {target_lengths.shape}")
        # print("__________________________________________________________________________")

        # Calculate the CTC loss
        # loss = ctc_loss(logits, targets, logit_lengths, target_lengths)
        loss = ce_loss_util(logits, targets)

        # print(loss.item())
        i += 1

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(crnn_model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dl)
    accuracy = tot_correct_tr / tot_samples_tr
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Validation
    # if (epoch + 1) % 1 == 0:
    crnn_model.eval()
    val_loss = 0.0
    val_tot_correct = 0
    val_tot_samples = 0

    with torch.no_grad():
        for sample in dl:
            val_images, val_targets = sample["image"], sample["label"]
            val_images = val_images.to(device)
            val_targets = val_targets.to(device)
            

            val_logits = crnn_model(val_images)
            val_loss = ce_loss_util(val_logits, val_targets)
            if epoch % 10 == 0:
                print(f'Val Loss: {avg_loss:.4f}')

torch.save(crnn_model.state_dict(), 'chk_pts/crnn_model.pth')
writer.close()