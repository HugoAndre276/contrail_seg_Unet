# %%

import numpy as np
import torch
from torch.utils.data import DataLoader

import data
from contrail import ContrailModel

torch.set_grad_enabled(False)


# %%
model1 = ContrailModel("UNet", in_channels=1, out_classes=1)
model1.load_state_dict(torch.load("data/models/google_fewshot_1000-dice-30minute.torch"))

model2 = ContrailModel("UNet", in_channels=1, out_classes=1)
model2.load_state_dict(torch.load("data/models/google_fewshot_2000-dice-30minute.torch"))

model3 = ContrailModel("UNet", in_channels=1, out_classes=1)
model3.load_state_dict(torch.load("data/models/google_fewshot_3000-dice-30minute.torch"))


# %%
train_dataset, test_dataset = data.own_dataset(train=False)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=3,
    num_workers=0,
    shuffle=True,
)

batch = next(iter(test_dataloader))

model1.eval()
logits1 = model1(batch[0])
pred1 = logits1.sigmoid()

model2.eval()
logits2 = model2(batch[0])
pred2 = logits2.sigmoid()

model3.eval()
logits3 = model3(batch[0])
pred3 = logits3.sigmoid()


for i in range(len(batch[0])):
    image = batch[0][i]
    labeled = batch[1][i]

    d = {
        "Image": np.array(image),
        "Labeled": np.array(labeled),
        "Google1000-dice-30minute": np.array(pred1[i]),
        "Google2000-dice-30minute": np.array(pred2[i]),
        "Google3000-dice-30minute": np.array(pred3[i]),
    }

    data.visualize(**d)
    

# %%
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1000,
    num_workers=0,
    shuffle=True,
)

batch = next(iter(test_dataloader))
        
model_names = ["Dice score for Google1000", "Dice score for Google2000", "Dice score for Google3000"]

with torch.no_grad():
    for i, model in enumerate([model1, model2, model3]):
        model.eval()
        logits = model(batch[0])
        predict = logits.sigmoid()
        target = batch[1]
        intersection = torch.nansum(predict * target)
        cardinality = torch.nansum(predict + target)
        dice = (2 * intersection) / cardinality 
        print(model_names[i], "\t", round(dice.item(), 5))
        

# %% 
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1000,
    num_workers=0,
    shuffle=True,
)

batch = next(iter(test_dataloader))
        
model_names = ["F1-score for Google1000", "F1-score for Google2000", "F1-score for Google3000"]

with torch.no_grad():
    for i, model in enumerate([model1, model2, model3]):
        model.eval()
        logits = model(batch[0])
        predict = logits.sigmoid()
        
        # Binarize predictions for F1 score calculation
        predict = (predict > 0.5).float()
        target = batch[1]

        # Calculate Precision and Recall
        true_positives = torch.sum(predict * target)
        predicted_positives = torch.sum(predict)
        actual_positives = torch.sum(target)

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        
        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        print(model_names[i], "\t", round(f1_score.item(), 5))