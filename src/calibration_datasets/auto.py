from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import CalibrationDatasetBase
from .registry import register_dataset


@register_dataset("auto")
class AutoHFDataset(CalibrationDatasetBase):
    def __init__(self, dataset_id: str, split: str, num_samples: int = 512, max_sequence_length: int = 2048) -> None:
        self.dataset_id = dataset_id
        self.split = split
        self.num_samples = num_samples
        self.max_sequence_length = max_sequence_length

    def get_dataset(self, tokenizer: PreTrainedTokenizerBase, seed: int = 42, **kwargs) -> Dataset:
        """Auto-load a dataset from Hugging Face (HF) Hub using the HF datasets library.

        Args:
            tokenizer (PreTrainedTokenizerBase): _description_
            num_samples (int, optional): _description_. Defaults to 512.
            max_sequence_length (int, optional): _description_. Defaults to 2048.
            seed (int, optional): Seed to use for dataset shuffling. Defaults to 42.
            **kwargs: Additional keyword arguments for the `load_dataset` method.

        Returns:
            Dataset | IterableDataset : A huggingface compatible dataset.
        """
        # Load the dataset
        ds = load_dataset(self.dataset_id, split=self.split, **kwargs)

        # Shuffle the samples and select subset of data
        assert isinstance(ds, Dataset), "Expected type Dataset. Ensure split is specified and streaming is False."
        ds = ds.shuffle(seed=seed).select(range(self.num_samples))

        # Map functions for preprocessing and tokenization
        def preprocess(example):
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

        ds = ds.map(preprocess)

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=self.max_sequence_length,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

        return ds
    
    @classmethod
    def __help__(cls) -> str:
        return "Any HuggingFace compatible dataset, either on HF hub or on disk. Pass the path or the model id."
