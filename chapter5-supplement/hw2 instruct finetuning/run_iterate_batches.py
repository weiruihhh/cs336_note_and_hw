def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    batches = []
    for i in range(0, len(dataset), batch_size):
        #不用再显式判断是否越界了，因为切片会自动处理越界的情况（如果越界了就切到最后），所以直接切片就好。
        batch_indices = indices[i:i+batch_size]
        batch = [dataset[j] for j in batch_indices]
    
        batch_dict = {"input_ids": torch.stack([b["input_ids"] for b in batch]), "labels": torch.stack([b["labels"] for b in batch])}
        batches.append(batch_dict)
    
    return batches
