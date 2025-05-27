import argparse
import copy
import random

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from model import FFN


if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=575)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=60)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--generate_every", type=int, default=4)
    parser.add_argument("--generate_length", type=int, default=50)
    parser.add_argument("--num_generate", type=int, default=10)
    parser.add_argument("--temp", type=float, default=2.5)
    parser.add_argument(
        "--train_data",
        type=str,
        default="/dropbox/24-25/574/data/sst/train-reviews.txt",
    )
    parser.add_argument(
        "--dev_data",
        type=str,
        default="/dropbox/24-25/574/data/sst/dev-reviews.txt",
    )
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build datasets
    sst_train = SSTLanguageModelingDataset.from_file(args.train_data)
    sst_dev = SSTLanguageModelingDataset.from_file(args.dev_data, vocab=sst_train.vocab)
    # arrays of entire dev set
    dev_data = sst_dev.batch_as_tensors(0, len(sst_dev))
    # convert to Tensors
    dev_data = {key: torch.LongTensor(value) for key, value in dev_data.items()}

    # build model
    padding_index = sst_train.vocab[SSTLanguageModelingDataset.PAD]
    # get the language model
    model = LSTMLanguageModel(
        args.embedding_dim,
        args.hidden_dim,
        len(sst_train.vocab),
        padding_index,
        args.dropout,
    )

    # get training things set up
    data_size = len(sst_train)
    batch_size = args.batch_size
    starts = list(range(0, data_size, batch_size))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2, lr=args.lr)
    best_loss = float("inf")
    best_model = None

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        # shuffle batches
        random.shuffle(starts)
        for start in tqdm(starts):
            batch = sst_train.batch_as_tensors(
                start, min(start + batch_size, data_size)
            )
            model.train()
            # get probabilities and loss
            # [batch_size, max_seq_len, vocab_size]
            logits = model(
                torch.LongTensor(batch["text"]), torch.LongTensor(batch["length"])
            )
            # transpose for torch cross entropy format
            # [batch_size, vocab_size, max_seq_len]
            logits = logits.transpose(1, 2)
            # [batch_size, max_seq_len]
            all_loss = torch.nn.functional.cross_entropy(
                logits,
                # batch["target"]: [batch_size, max_seq_len]
                torch.LongTensor(batch["target"]),
                reduction="none",
            )
            # mask out the PAD symbols in the loss
            loss = mask_loss(all_loss, batch["target"], padding_index)

            running_loss += loss.item()

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} train loss: {running_loss / len(starts)}")

        # get dev loss every epoch
        model.eval()
        # [batch_size, max_seq_len, vocab_size]
        logits = model(dev_data["text"], dev_data["length"])
        epoch_loss = mask_loss(
            torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), dev_data["target"], reduction="none"
            ),
            dev_data["target"].numpy(),
            padding_index,
        )
        print(
            f"Epoch {epoch} dev loss: {epoch_loss.item()}; perplexity (nats): {epoch_loss.exp()}"
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New best loss; saving current model")
            best_model = copy.deepcopy(model)

        # generate some text every N epochs
        if (epoch + 1) % args.generate_every == 0:
            print(
                generate(
                    model,
                    sst_train.vocab[SSTLanguageModelingDataset.BOS],
                    args.num_generate,
                    args.generate_length,
                    sst_train.vocab,
                    temp=args.temp,
                )
            )

