import config
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import torch
import dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertBaseUncase
import engine



def run():
    df = pd.read_csv(config.TRAIN_FILE).fillna("none")
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    train_df, test_df = train_test_split(
        df,
        test_size = 0.99,
        stratify= df.sentiment.values,
        random_state = 42
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = dataset.BertDataset(
        reviews=train_df.review.values,
        targets=train_df.sentiment.values
    )

    test_dataset = dataset.BertDataset(
        reviews=test_df.review.values,
        targets=test_df.sentiment.values
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE
        #num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE
    )

    device = torch.device("cpu")
    model = BertBaseUncase()
    model.to(device)

    num_training_steps = int(len(train_df)//config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train(train_loader, model, optimizer, device, scheduler)
        torch.save(model.state_dict(), config.MODEL_PATH)
        """
        tar, pred = engine.eval(test_loader, model, device)
        tar = tar.view(1, -1)
        pred = np.array(pred) >= 0.5
        accuracy = accuracy_score(tar, pred)
        if best_accuracy < accurcy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
        """




if __name__ == "__main__":
    run()