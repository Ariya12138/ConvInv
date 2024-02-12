import os
from typing import Any, Dict

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer,RobertaForSequenceClassification,BertModel,AutoTokenizer
import torch.nn.functional as F

EMBEDDER_MODEL_NAMES = [
    "bert",
    "contriever",
    "dpr",
    "gtr_base",
    "gtr_base__random_init",
    "medicalai/ClinicalBERT",
    "gtr_large",
    "ance_tele",
    "dpr_st",
    "gtr_base_st",
    "paraphrase-distilroberta",
    "sentence-transformers/all-MiniLM-L6-v2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "ANCE",
    "BGE"
]


FREEZE_STRATEGIES = ["decoder", "encoder_and_decoder", "encoder", "none"]
EMBEDDING_TRANSFORM_STRATEGIES = ["repeat"]


def get_device():
    """
    Function that checks
    for GPU availability and returns
    the appropriate device.
    :return: torch.device
    """
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device


device = get_device()


def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()
    # print(f"Froze {total_num_params} params from model type {type(model)}")


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def max_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.max(dim=1).values
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def stack_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.reshape((B, S * D))  # stack along seq length
    assert pooled_outputs.shape == (B, S * D)
    return pooled_outputs

class BGE(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_parallel = False
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state = super().forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = last_hidden_state[:, 0]
        output = F.normalize(output, p=2, dim=1)
        return output
    
# ANCE model
class ANCE(RobertaForSequenceClassification):
    # class Pooler:   # adapt to DPR
    #     def __init__(self, pooler_output):
    #         self.pooler_output = pooler_output

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)  # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)

def load_embedder_and_tokenizer(name: str, torch_dtype: str):
    # TODO make abstract/argparse for it etc.
    # name = "gpt2" #### <--- TEMP. For debugging. Delete!
    model_kwargs = {
        "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
        "output_hidden_states": False,
    }

    if name == "dpr":
        # model = SentenceTransformer("sentence-transformers/facebook-dpr-question_encoder-multiset-base")
        model = transformers.DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
    elif name == "dpr_st":
        # TODO figure out why model w/ sentence transformers gives different results.
        model = SentenceTransformer(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
        )
        tokenizer = model.tokenizer
    elif name == "contriever":
        model = transformers.AutoModel.from_pretrained(
            "facebook/contriever", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever")
    elif name == "bert":
        model = transformers.AutoModel.from_pretrained(
            "bert-base-uncased", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    elif name == "gtr_base":
        model = transformers.AutoModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/gtr_t5_base", **model_kwargs
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/home/yiruo_cheng/PLMs/gtr_t5_base"
        )
    elif name == "gtr_base__random_init":
        config = transformers.AutoConfig.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        model = transformers.AutoModel.from_config(config).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "gtr_base_st":
        print("!!!!!!!!!!!!!!!!!!!gtr_base_st!!!!!!!!!!!!!!!!!!")
        model = SentenceTransformer("/home/yiruo_cheng/PLMs/sentence-transformers_gtr-t5-base")
        tokenizer = model.tokenizer
    elif name == "ANCE":
        print("!!!!!!!!!!!!!!!!!!!!!!ANCE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        ance_model_path = "/home/yiruo_cheng/PLMs/ance-msmarco"
        config = RobertaConfig.from_pretrained(ance_model_path,finetuning_task="MSMarco")
        tokenizer = RobertaTokenizer.from_pretrained( ance_model_path,do_lower_case=True)
        model = ANCE.from_pretrained(ance_model_path, config=config)
    
    elif name == "BGE":
        print("!!!!!!!!!!!!!!!!!!!!!!BGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        bge_model_path = "/home/yiruo_cheng/PLMs/models--BAAI--bge-base-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(bge_model_path,do_lower_case=True)
        model = BGE.from_pretrained(bge_model_path)
    
    elif name == "gtr_large":
        model = SentenceTransformer("sentence-transformers/gtr-t5-large")
        tokenizer = model.tokenizer
    elif name == "ance_tele":
        model = transformers.AutoModel.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "OpenMatch/ance-tele_nq_psg-encoder"
        )
    elif name == "paraphrase-distilroberta":
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-distilroberta-base-v1"
        )
    elif name == "medicalai/ClinicalBERT":
        model = transformers.AutoModel.from_pretrained(
            "medicalai/ClinicalBERT", **model_kwargs
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    elif name.startswith("gpt2"):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
        )
        # model.to_bettertransformer()
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("meta-llama/"):
        if torch_dtype == "float32":
            torch_dtype = torch.float32
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
            token=os.environ.get("LLAMA_TOKEN"),
            torch_dtype=torch_dtype,
        )
        # if torch_dtype is not torch.float32:
        #     model.to_bettertransformer()
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("sentence-transformers/"):
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer
    else:
        print(f"WARNING: Trying to initialize from unknown embedder {name}")
        model = transformers.AutoModel.from_pretrained(name, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    # model = torch.compile(model)
    return model, tokenizer


def load_encoder_decoder(
    model_name: str, lora: bool = False
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }
    if lora:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": "auto",
            }
        )
    if model_name == "t5-base":
        return transformers.AutoModelForSeq2SeqLM.from_pretrained("/home/yiruo_cheng/PLMs/t5-base")
    else:
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **model_kwargs
        )


def load_tokenizer(name: str, max_length: int) -> transformers.PreTrainedTokenizer:
    if name == "t5-base":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/home/yiruo_cheng/PLMs/t5-base",
            padding=True,
            truncation="max_length",
            max_length=max_length,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            name,
            padding=True,
            truncation="max_length",
            max_length=max_length,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable super annoying warning:
    # https://github.com/huggingface/transformers/issues/22638
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer
