from base import CalibrationDatasetBase
from datasets import Dataset, IterableDataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class UltraChat200k(CalibrationDatasetBase):
    @classmethod
    def get_dataset(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int = 512,
        max_sequence_length: int = 2048,
        # seed: int = 42,
        **kwargs,
    ) -> Dataset | IterableDataset:
        """The fine_tuning split of the ultrachat 200k dataset

        Args:
            tokenizer (PreTrainedTokenizerBase): A huggingface compatible tokenizer class.
            num_samples (int): The number of samples from the dataset to use.
            max_sequence_length (int): The maximum sequence length.
            seed (int): The random seed to use when shuffling the dataset.
            **kwargs (Any): keyword arguments for the huggingface datasets.load_dataset method. Split arg is already
                defined.

        Returns:
            Dataset | IterableDataset : A huggingface compatible dataset.
        """
        # Load and preprocess the ultrachat_200k dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", **kwargs)
        ds = ds.shuffle(seed=42)
        if isinstance(ds, Dataset):
            ds = ds.select(range(num_samples))
        elif isinstance(ds, IterableDataset):
            ds = ds.take(num_samples)
        else:
            raise TypeError(
                f"Was expecting either a Dataset or IterableDataset but got {type(ds)}. Make sure the split is defined"
            )

        def preprocess(example):
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

        ds = ds.map(preprocess)

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

        return ds
