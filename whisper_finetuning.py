from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# import the relavant libraries for loggin in
from huggingface_hub import HfApi, HfFolder

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions and procedures
def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)

    return None


def prepare_dataset(batch):
    """
    Prepare audio data to be suitable for Whisper AI model.
    """
    # (1) load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # (2) compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    """
    Define evaluation metrics. We will use the Word Error Rate (WER) metric.
    For more information, check:
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# STEP 0. Loging to Hugging Face
# get your account token from https://huggingface.co/settings/tokens
token = 'PASTE_YOUR_HUGGING_FACE_TOKEN_HERE'
login_hugging_face(token)
print('We are logged in to Hugging Face now!')


# STEP 1. Download Dataset
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "lt", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "lt", split="test", use_auth_token=True)

common_voice = common_voice.remove_columns(
    ["accent",
     "age",
     "client_id",
     "down_votes",
     "gender",
     "locale",
     "path",
     "segment",
     "up_votes"]
    )

print(common_voice)

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 2. Prepare: Feature Extractor, Tokenizer and Data
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

# - Load Feature extractor: WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")

# - Load Tokenizer: WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="Lithuanian", task="transcribe")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 3. Combine elements with WhisperProcessor
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="Lithuanian", task="transcribe")

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 4. Prepare Data
print('| Check the random audio example from Common Voice dataset to see what form the data is in:')
print(f'{common_voice["train"][0]}\n')

# -> (1): Downsample from 48kHZ to 16kHZ
from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print('| Check the effect of downsampling:')
print(f'{common_voice["train"][0]}\n')

# Prepare and use function to prepare our data ready for the Whisper AI model
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=2 # num_proc > 1 will enable multiprocessing
    )

# - - - - - - - - - - - - - - - - - - - - - |
# STEP 5. Training and evaluation
# STEP 5.1. Initialize the Data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# STEP 5.1. Define evaluation metric
import evaluate
metric = evaluate.load("wer")

# STEP 5.3. Load a pre-trained Checkpoint
from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

"""
Overide generation arguments:
- no tokens are forced as decoder outputs: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids
- no tokens are suppressed during generation: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens
"""
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# STEP 5.4. Define the training configuration
"""
Check for Seq2SeqTrainingArguments here:
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
"""
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-lt",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # testing
)


# Initialize a trainer.
"""
Forward the training arguments to the Hugging Face trainer along with our model,
dataset, data collator and compute_metrics function.
"""
from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Save processor object before starting training
processor.save_pretrained(training_args.output_dir)

# STEP 5.5. Training
"""
Training will take appr. 5-10 hours depending on your GPU.
"""
print('Training is started.')
trainer.train()  # <-- !!! Here the training starting !!!
print('Training is finished.')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
We can submit our checkpoint to the hf-speech-bench on push by setting the
appropriate key-word arguments (kwargs):
https://huggingface.co/spaces/huggingface/hf-speech-bench
"""

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: lt, split: test",
    "language": "lt",
    "model_name": "Whisper Large LT - Vytautas Bielinskas",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-large",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

# Upload training results to the Hub.
trainer.push_to_hub(**kwargs)
print('Trained model uploaded to the Hugging Face Hub')
