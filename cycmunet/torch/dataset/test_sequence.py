import unittest
import pathlib
from PIL import Image
from ..dataset import ImageSequenceDataset  # Assuming the class is in a module named dataset   


class TestImageSequenceDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary index file and dummy images for testing
        self.index_file = "test_index.txt"
        self.dataset_dir = pathlib.Path("test_dataset")
        self.dataset_dir.mkdir(exist_ok=True)
        (self.dataset_dir / "sequences").mkdir(exist_ok=True)

        # Create dummy image sequences
        sequence_dir = self.dataset_dir / "sequences" / "seq1"
        sequence_dir.mkdir(exist_ok=True)
        for i in range(5):
            img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img.save(sequence_dir / f"img_{i}.png")

        # Write index file
        with open(self.index_file, "w", encoding="utf-8") as f:
            f.write("seq1\n")

        # Initialize the dataset
        self.dataset = ImageSequenceDataset(
            index_file=self.index_file,
            patch_size=(32, 32),
            scale_factor=2,
            augment=True,
            seed=42
        )

    def tearDown(self):
        # Clean up temporary files
        for file in self.dataset_dir.rglob("*"):
            file.unlink()
        self.dataset_dir.rmdir()
        pathlib.Path(self.index_file).unlink()

    def test_len(self):
        self.assertEqual(len(self.dataset), 1)

    def test_getitem(self):
        original,