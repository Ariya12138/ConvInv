import sklearn
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/evaluation')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import copy
from typing import List
import torch
import numpy as np
import torch
import transformers

import vec2text
from vec2text.models.model_utils import device
from transformers import DisjunctiveConstraint, PhrasalConstraint
from sentence_transformers import SentenceTransformer
from models_conv import load_model


SUPPORTED_MODELS = ["text-embedding-ada-002","text-embedding-gtr_st","text-embedding-ance","text-embedding-bge"]

def load_corrector(embedder: str) -> vec2text.trainers.Corrector:
    """Gets the Corrector object for the given embedder.

    For now, we just support inverting OpenAI Ada 002 embeddings; we plan to
    expand this support over time.
    """
    assert (
        embedder in SUPPORTED_MODELS
    ), f"embedder to invert `{embedder} not in list of supported models: {SUPPORTED_MODELS}`"

    if embedder == "text-embedding-ada-002":
        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer"
        )
        model = vec2text.models.CorrectorEncoderModel.from_pretrained(
            "jxm/vec2text__openai_ada002__msmarco__msl128__corrector"
        )
    elif embedder == "text-embedding-gtr_st":
        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/vec2text_gtr_st_msl128_inversion_100epochs"
        )
        model = vec2text.models.CorrectorEncoderModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/vec2text_gtr_st_msl48_corrector_100epochs_100inversion"
        )
    elif embedder == "text-embedding-ance":
        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/ance_msl48_inversion_50epochs_best"
        )
        model = vec2text.models.CorrectorEncoderModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/ance_msl48_corrector_80epochs_best"
        )
    elif embedder == "text-embedding-bge":
        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/bge_msl48_inversion_50epochs"
        )
        model = vec2text.models.CorrectorEncoderModel.from_pretrained(
            "/home/yiruo_cheng/PLMs/bge_msl48_corrector_100epochs"
        )

    # print(inversion_model.device)
    # print(model)
    inversion_model.to('cuda')
    model.to('cuda')
    
    # print(inversion_model.device)
    # print(model.device)
    
    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )

    # backwards compatibility stuff
    model.config.dispatch_batches = None
    corrector = vec2text.trainers.Corrector(
        model=model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    
    return corrector


def invert_embeddings(
    embeddings: torch.Tensor,
    corrector: vec2text.trainers.Corrector,
    num_steps: int = None,
    sequence_beam_width: int = 0,
    hypothesis_input_ids: torch.Tensor = None,
    hypothesis_attention_mask: torch.Tensor = None,
    hypothesis_embedding: torch.Tensor = None,
    initial_hypothesis_str: str = None,
) -> List[str]:
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()

    gen_kwargs = copy.copy(corrector.gen_kwargs)
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 48

    
    
    '''
    #gen_kwargs["constraints"] = [PhrasalConstraint([corrector.tokenizer("What").input_ids[0]])]
    constraint_What = [corrector.tokenizer("What").input_ids[0]]
    constraint_what = [corrector.tokenizer("what").input_ids[0]]
    

    constraints_disjunctive = [DisjunctiveConstraint([constraint_What,constraint_what])]
    gen_kwargs["constraints"] = constraints_disjunctive
    '''
    #gen_kwargs["repetition_penalty"] = 10.0
    if num_steps is None:
        assert (
            sequence_beam_width == 0
        ), "can't set a nonzero beam width without multiple steps"

        regenerated = corrector.inversion_trainer.generate(
            inputs={
                "frozen_embeddings": embeddings,
            },
            generation_kwargs=gen_kwargs,
        )
    elif hypothesis_input_ids != None:
        
        corrector.return_best_hypothesis = sequence_beam_width > 0
        regenerated = corrector.generate(
            inputs={
                "frozen_embeddings": embeddings,
                "hypothesis_input_ids":hypothesis_input_ids,
                "hypothesis_attention_mask":hypothesis_attention_mask,
                "hypothesis_embedding":hypothesis_embedding
            },
            generation_kwargs=gen_kwargs,
            num_recursive_steps=num_steps,
            sequence_beam_width=sequence_beam_width,
        )
        
    else:
        #print("initial_hypothesis_str:",initial_hypothesis_str)
        corrector.return_best_hypothesis = sequence_beam_width > 0
        regenerated = corrector.generate(
            inputs={
                "frozen_embeddings": embeddings,
                #"initial_hypothesis_str": initial_hypothesis_str,
            },
            generation_kwargs=gen_kwargs,
            num_recursive_steps=num_steps,
            sequence_beam_width=sequence_beam_width,
        )
    
    output_strings = corrector.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings


def invert_strings(
    strings: List[str],
    corrector: vec2text.trainers.Corrector,
    num_steps: int = None,
    sequence_beam_width: int = 0,
) -> List[str]:
    inputs = corrector.embedder_tokenizer(
        strings,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        frozen_embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    return invert_embeddings(
        embeddings=frozen_embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )



def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask



if __name__ == '__main__':
    
    corrector = vec2text.load_corrector("text-embedding-gtr_st")
    logger.info("inversion,corrector and tokenizer is loaded!")
    sentence1 = "Is it treatable?"
    sentence2 = "What is throat cancer?"
    sentence3 = "Let's play basketball"
    sentence4 = "I want to sleep. I don't want to go to school!"
    max_query_length = 32
    model_checkpoint_path = "/home/yiruo_cheng/proposal/CDRvec2text/experiments/train_kd_for_inversion/qrecc/gtr-conv/checkpoints/epoch-1"
    query_tokenizer, query_encoder = load_model("gtr-base", "query", model_checkpoint_path)
    query_encoder = query_encoder.to("cuda")
    inversion_utt = query_tokenizer.encode(sentence1, add_special_tokens = True, max_length = max_query_length)
    print(inversion_utt)
    inversion, mask_inversion = padding_seq_to_same_length(inversion_utt, max_pad_length = max_query_length)
    inversion = torch.tensor(inversion).unsqueeze(0).to("cuda")
    mask_inversion = torch.tensor(mask_inversion).unsqueeze(0).to("cuda")
    
    query_embs = query_encoder(inversion, mask_inversion)
    query_embeddings1 = query_embs.detach()

    inversion_utt = query_tokenizer.encode(sentence2, add_special_tokens = True, max_length = max_query_length)
    inversion, mask_inversion = padding_seq_to_same_length(inversion_utt, max_pad_length = max_query_length)
    inversion = torch.tensor(inversion).unsqueeze(0).to("cuda")
    mask_inversion = torch.tensor(mask_inversion).unsqueeze(0).to("cuda")
    
    query_embs = query_encoder(inversion, mask_inversion)
    query_embeddings2 = query_embs.detach()

    result_tensor = torch.cat((query_embeddings1, query_embeddings2), dim=0)
    print(result_tensor.shape)
    
    """
    model = SentenceTransformer("/home/bingxing2/home/scx6964/yiruo_cheng/PLMs/sentence-transformers_gtr-t5-base")
    tokenizer = model.tokenizer
    model = model.cuda()
    input_ids = tokenizer.encode(sentence1, sentence2, add_special_tokens=True)
   


    sentence = f"{sentence1}{sentence2}"
    # model_path = "/home/yiruo_cheng/PLMs/sentence-transformers_gtr-t5-base"
    print(sentence)
    """
    result_tensor = result_tensor.to("cuda")
    test = vec2text.invert_embeddings(result_tensor,corrector=corrector,num_steps=20,sequence_beam_width=4)
    for sentence in test:
        print(sentence)
    


    """
    model = SentenceTransformer("/home/bingxing2/home/scx6964/yiruo_cheng/PLMs/sentence-transformers_gtr-t5-base")
    tokenizer = model.tokenizer
    model = model.cuda()

    test1 = "[CLS]Tell me about lung cancer.[SEP]Is it treatable?[SEP]What is throat cancer?[SEP]"
    test2 = "[CLS]Tell me about lung cancer.[SEP]What is throat cancer?[SEP]Is it treatable?[SEP]"

    with torch.no_grad():
        embedding1 = model.encode(test1)
        embedding2 = model.encode(test2)
    sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    # test = vec2text.invert_strings(sentence,corrector=corrector,num_steps=20,sequence_beam_width=4)
    print(sim)
    """
 
