from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import torch

tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo/vocab.json",
    "./EsperBERTo/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

NUM_HEADS = 5
VEC_LEN = 30
DUPLICATE_OUTPUTS = 4

from transformers_local.src.transformers.configuration_roberta import RobertaConfig

roberta_config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    hidden_size=VEC_LEN*DUPLICATE_OUTPUTS*NUM_HEADS,
    num_attention_heads=NUM_HEADS,
    num_hidden_layers=6,
    type_vocab_size=1,
)

standard_config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    hidden_size=540,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from NeuronalTransformerLayers.NeuronBankConfig import NeuronBankConfig
neuron_config = NeuronBankConfig(
    query_len=VEC_LEN+1,
    values_len=VEC_LEN,
    num_heads=NUM_HEADS,
    num_duplicates=DUPLICATE_OUTPUTS,
    num_neurons=500,
)


from transformers_local.src.transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

from transformers_local.src.transformers.modeling_roberta import RobertaForMaskedLM, NeuralRobertaForMaskedLM

USE_NEURONAL = True

if USE_NEURONAL:
    model = NeuralRobertaForMaskedLM(roberta_config, neuron_config)
else:
    model = RobertaForMaskedLM(standard_config)

print(model.num_parameters())

from transformers_local.src.transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)

from transformers_local.src.transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers_local.src.transformers import Trainer, TrainingArguments

import logging

logging.basicConfig(level=logging.INFO)

training_args = TrainingArguments(
    output_dir='./EsperBERTo' + ('_neuronal' if USE_NEURONAL else '_standard'),
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=8,
    save_steps=10_000,
    logging_steps=10_000,
    #warmup_steps=30_000,
    logging_dir='neuronal_log' if USE_NEURONAL else 'standard_log',
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()