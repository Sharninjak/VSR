import dataset

dataset_types = {
    'triplet': dataset.ImageSequenceDataset,
    'video': dataset.VideoFrameDataset
}

print(dataset_types.types())
