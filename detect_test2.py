#  %%
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

import data
from contrail import ContrailModel

torch.set_grad_enabled(False)


# %%
model = ContrailModel("UNet", in_channels=1, out_classes=1)
model.load_state_dict(torch.load("data/models/google_fewshot_3000-dice-30minute.torch"))

# %%
train_dataset, test_dataset = data.own_dataset(train=False)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=10,
    num_workers=0,
    shuffle=True,
)

batch = next(iter(test_dataloader))
models = batch[0]
images = batch[1]

model_name = "AUC score for Google3000"

with torch.no_grad():
    model.eval()
    logits = model(batch[0])
    predict = logits.sigmoid()
    target = batch[1]

    # De-binarize predictions for AUC score calculation
    predict = predict.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    # Binarize predictions for F1 score calculation
    target = (target > 0.5).astype(np.uint8)
    
    # Calculate the AUC score with scikit learn
    auc_score = roc_auc_score(target,predict)

    print(model_name, "\t", round(auc_score, 3))