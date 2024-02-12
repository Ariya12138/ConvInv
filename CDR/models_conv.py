from IPython import embed
import sys

sys.path += ['../']
import numpy as np

import torch
from torch import nn
import transformers
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          AutoTokenizer, AutoModel, BertModel,
                          DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
                          DPRContextEncoder, DPRQuestionEncoder, T5EncoderModel, T5Config, T5PreTrainedModel)

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb
    

class GTR(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # T5EncoderModel.__init__(self, config)
        self.t5_encoder = T5EncoderModel(config)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size, bias=False) # gtr has
        self.activation = torch.nn.Identity()
        self.model_parallel = False
        
    def pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        output = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.pooling(output, attention_mask)    
        output = self.activation(self.embeddingHead(output))
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


# TCTColBERT model
class TCTColBERT(nn.Module):
    def __init__(self, model_path) -> None:
        super(TCTColBERT, self).__init__()
        self.model = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if "cur_utt_end_position" in kwargs:
            device = outputs.device
            cur_utt_end_positions = kwargs["cur_utt_end_positions"]
            output_mask = torch.zeros(attention_mask.size()).to(device)
            mask_row = []
            mask_col = []
            for i in range(len(cur_utt_end_positions)):
                mask_row += [i] * (cur_utt_end_positions[i] - 3)
                mask_col += list(range(4, cur_utt_end_positions[i] + 1))

            mask_index = (
                torch.tensor(mask_row).long().to(device),
                torch.tensor(mask_col).long().to(device)
            )
            values = torch.ones(len(mask_row)).to(device)
            output_mask = output_mask.index_put(mask_index, values)
        else:
            output_mask = attention_mask
            output_mask[:, :4] = 0  # filter the first 4 tokens: [CLS] "[" "Q/D" "]"

        # sum / length
        sum_outputs = torch.sum(outputs * output_mask.unsqueeze(-1), dim=-2)
        real_seq_length = torch.sum(output_mask, dim=1).view(-1, 1)

        return sum_outputs / real_seq_length


'''
Model-related functions
'''
class BGE(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_parallel = False
    
    def forward(self, input_ids, attention_mask):
        last_hidden_state = super().forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output = last_hidden_state[:, 0]
        output = F.normalize(output, p=2, dim=1)
        return output


def load_model(model_type, query_or_doc, model_path):
    assert query_or_doc in ("query", "doc")
    if model_type.lower() == "ance":
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE.from_pretrained(model_path, config=config)
    elif model_type.lower() == "dpr-nq":
        if query_or_doc == "query":
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path)
            model = DPRQuestionEncoder.from_pretrained(model_path)
        else:
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_path)
            model = DPRContextEncoder.from_pretrained(model_path)
    elif model_type.lower() == "tctcolbert":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TCTColBERT(model_path)
    elif model_type.lower() == "gtr-base":
        tokenizer = AutoTokenizer.from_pretrained(model_path,do_lower_case=True)
        model = GTR.from_pretrained(model_path)
    elif model_type.lower() == "bge":
        print("Loading BGE model")
        tokenizer = AutoTokenizer.from_pretrained(model_path,do_lower_case=True)
        model = BGE.from_pretrained(model_path)
    else:
        raise ValueError

    # tokenizer.add_tokens(["<CUR_Q>", "<CTX>", "<CTX_R>", "<CTX_Q>"])
    # model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

'''
sentences = ["This is an example sentence", "Each sentence is converted"]
model_type = "gtr-base"
model_path = "/home/yiruo_cheng/PLMs/gtr-t5-base"
tokenizer,model = load_model(model_type,"query",model_path)
input_encodings = tokenizer(sentences, padding=True, return_tensors='pt')

output = model(**input_encodings)
print(type(output))

model2 = SentenceTransformer("/home/yiruo_cheng/PLMs/sentence-transformers_gtr-t5-base")
output2 = model2.encode(sentences)
print(type(output2))
'''