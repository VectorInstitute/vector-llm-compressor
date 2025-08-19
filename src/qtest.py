from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load and preprocess the dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
if isinstance(ds, Dataset):
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
else:
    raise ValueError()
 
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithms
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model: Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
