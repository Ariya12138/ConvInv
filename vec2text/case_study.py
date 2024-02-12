import sys
import torch
from transformers import AutoTokenizer
sys.path.append('..')
sys.path.append('.')
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/convRetriever')
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/evaluation')
from models_conv import load_model
from convsearch_dataset import padding_seq_to_same_length

from api import load_corrector,invert_embeddings




def get_session_concat(cur_utt_text,ctx_utts_text,last_response_text=None):
    model_checkpoint_path = "/home/yiruo_cheng/PLMs/kdGTR"
    query_tokenizer, query_encoder = load_model("gtr-base", "query", model_checkpoint_path)
    query_encoder = query_encoder.to("cuda")
    full_text = cur_utt_text
    if last_response_text != None:
        full_text += last_response_text
    for j in range(len(ctx_utts_text) - 1, -1, -1):
                full_text += ctx_utts_text[j]
    flat_concat = query_tokenizer.encode(full_text, add_special_tokens=True,max_length=512) # not remove [CLS]
    concat, concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = 512)
    concat = torch.tensor(concat).unsqueeze(0).to("cuda")
    concat_mask = torch.tensor(concat_mask).unsqueeze(0).to("cuda")
    session_embs = query_encoder(concat, concat_mask)
    session_embs = session_embs.detach().cpu()
    return session_embs


if __name__ == '__main__':
    
    corrector = load_corrector("text-embedding-gtr_st")
    print("inversion,corrector and tokenizer is loaded!")

    
    rewrite = ["What's the difference in throat cancer and esophageal cancer's symptoms?"]
    rewrite2 = "I want to eat apple"
    cur_utt_text = "What's the difference in their symptoms?"
    ctx_utts_text = ['What is throat cancer?', 'Is it treatable?', 'Tell me about lung cancer.', 'What are its symptoms? ', 'Can it spread to the throat?', 'What causes throat cancer?', 'What is the first sign of it?', 'Is it the same as esophageal cancer?']
    last_response_text = "Data collection from the test road is the most important activity. Test road (e.g. city, highway, etc.) measured data are the inputs to the 'Drive Cycle' preparation activity."
    session_embedding = get_session_concat(cur_utt_text,ctx_utts_text,last_response_text).to("cuda")

    cur_utt_text = "I want to eat apple"
    ctx_utts_text = ["Let's play basketball"]
    last_response_text = "dvgfghsrftgn"

    #session_embedding2 = get_session_concat(cur_utt_text,ctx_utts_text,last_response_text).to("cuda")
    #session_embedding = torch.cat((session_embedding1, session_embedding2), dim=0)
    print(session_embedding.shape)
    # print(rewrite_embedding.shape)
    #rewrite = [rewrite1,rewrite2]
    
   

    
    

    
    texts = invert_embeddings(embeddings = session_embedding,
                                corrector = corrector,
                                num_steps = 20,
                                sequence_beam_width = 5,
                                initial_hypothesis_str = rewrite)
    
    print(texts)
    