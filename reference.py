class SSTClassificationDataset(Dataset):

    labels_to_string = {0: "terrible", 1: "bad", 2: "so-so", 3: "good", 4: "excellent"}
    label_one_hots = np.eye(len(labels_to_string))
    PAD = "<PAD>"

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        example = self.__getitem__(index)
        return {
            "review": np.array(self.vocab.tokens_to_indices(example["review"])),
            "label": example["label"],
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        examples = [self.example_to_tensors(index) for index in range(start, end)]
        padding_index = self.vocab[SSTClassificationDataset.PAD]
        return {
            "review": pad_batch(
                [example["review"] for example in examples], padding_index
            ),
            "label": np.array([example["label"] for example in examples]),
            "lengths": np.array([len(example["review"]) for example in examples]),