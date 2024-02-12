from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import json
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import transformers
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pprint import pprint
from os.path import join as oj

import sys
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/evaluation')
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/convRetriever')

from models_conv import load_model
from utils_conv import set_seed,json_dumps_arguments,check_dir_exist_or_build
from convsearch_dataset import QReCCDataset, CAsTDataset, TopiOCQADataset,padding_seq_to_same_length
from api import load_corrector,invert_embeddings


def get_rewrite_embedding(args):
    query_tokenizer, query_encoder = load_model(args.model_type, "query", args.rewrite_checkpoint_path)
    query_encoder = query_encoder.to(args.device)
    
    with open(args.T5QR_rewrite_file_path, 'r') as f:
        data = json.load(f)
    
    query_encoding_dataset = []
    for record in data:
        sample_id = record['sample_id']
        if args.use_T5QR_rewrite:
            print("use T5QR rewrite")
            query = record["t5_rewrite"]
            print(query)
        elif args.use_human_rewrite:
            print("use human rewrite")
            query = record["oracle_rewrite"]
            print(query)
        print("query: ",query)
        query_encoding_dataset.append([sample_id, query])

    def query_encoding_collate_fn(batch):
        bt_sample_ids, bt_src_seq = list(zip(*batch)) # unzip
        bt_src_encoding = query_tokenizer.batch_encode_plus(bt_src_seq,add_special_tokens = True, max_length = args.max_query_length)
        #inp = torch.tensor(bt_src_encoding.input_ids)
        bt_input_ids = bt_src_encoding.input_ids
        bt_attention_mask = bt_src_encoding.attention_mask
       
        return {"bt_sample_ids": bt_sample_ids, 
                "bt_input_ids":bt_input_ids, 
                "bt_attention_mask":bt_attention_mask}

    test_loader = DataLoader(query_encoding_dataset, 
                             batch_size = args.eval_batch_size, 
                             shuffle=False, 
                             collate_fn=query_encoding_collate_fn)
    
    query_encoder.zero_grad()
    qid2rewrite_input_ids = {}
    qid2rewrite_attention_mask ={}
    qid2rewrite_embedding = {}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            query_encoder.eval()
            bt_sample_ids = batch["bt_sample_ids"]
            bt_input_ids = batch["bt_input_ids"]
            

            for i in range(len(bt_sample_ids)):
                if args.dataset == "cast20":
                    parts = bt_sample_ids[i].split('-')[1].split('_')
                    if len(parts) >= 2:
                        sample_id = '_'.join(parts[1:])
                    else:
                        sample_id = bt_sample_ids[i]
                else:
                    sample_id = bt_sample_ids[i]

                example_input_ids = bt_input_ids[i]
                
                input_ids,attention_mask = padding_seq_to_same_length(example_input_ids,max_pad_length = args.max_query_length)
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(args.device)
                attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(args.device)
                query_embs = query_encoder(input_ids, attention_mask)
                query_embs = query_embs.detach().cpu()
                
                



                # qid2query[sample_id] = query_embs[i].reshape(1, -1)
                qid2rewrite_input_ids[sample_id] = input_ids.detach().cpu()
                qid2rewrite_attention_mask[sample_id] = attention_mask.detach().cpu()
                qid2rewrite_embedding[sample_id] = query_embs
                # 将第 i 个查询嵌入向量重新形状为一个包含单一行和自动确定列数的二维数组

    
    torch.cuda.empty_cache()
    return qid2rewrite_input_ids,qid2rewrite_attention_mask,qid2rewrite_embedding
    


def get_query_embedding(args):
    set_seed(args)
    query_tokenizer,query_encoder = load_model(args.model_type, "query", args.model_checkpoint_path)
    query_encoder = query_encoder.to(args.device)
    doc_tokenizer = None
    
    # test dataset/dataloader
    logger.info("Buidling test dataset...")
    Datasets = {
        "qrecc": QReCCDataset,
        "cast19": CAsTDataset,
        "cast20": CAsTDataset,
        "cast21": QReCCDataset,
        "topiocqa": TopiOCQADataset
    }
    test_dataset = Datasets[args.dataset](args, query_tokenizer, doc_tokenizer, args.test_file_path, need_doc_info=False)
    test_loader = DataLoader(test_dataset, 
                            batch_size = args.eval_batch_size, 
                            shuffle=False, 
                            collate_fn=test_dataset.get_collate_fn(args))

    logger.info("Generating query embeddings for testing...")
    query_encoder.zero_grad()

    # qid2query = {}
    qid2query_inversion = {}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            query_encoder.eval()
            batch_sample_id = batch["bt_sample_id"]
            
            # test type
            if args.input_type == "oracle":
                input_ids = batch["bt_oracle_utt"].to(args.device)
                input_masks = batch["bt_oracle_utt_mask"].to(args.device)
            elif args.input_type == "raw":
                input_ids = batch["bt_cur_utt"].to(args.device)
                input_masks = batch["bt_cur_utt_mask"].to(args.device)
            elif args.input_type == "flat_concat":
                input_ids = batch["bt_concat"].to(args.device)
                input_masks = batch["bt_concat_mask"].to(args.device)
            elif args.input_type == "concat_for_inversion":
                # input_ids = batch["bt_concat"].to(args.device)
                # input_masks = batch["bt_concat_mask"].to(args.device)
                input_ids_inversion = batch["bt_concat_inversion"].to(args.device)
                input_masks_inversion = batch["bt_concat_mask_inversion"].to(args.device)
 
            else:
                raise ValueError("test input type:{}, has not been implemented.".format(args.test_input_type))


            # query_embs = query_encoder(input_ids, input_masks)
            # query_embs = query_embs.detach().cpu().numpy()
            
            query_embs_inversion = query_encoder(input_ids_inversion, input_masks_inversion)
            query_embs_inversion = query_embs_inversion.detach().cpu()
            
    
            for i in range(len(batch_sample_id)):
                if args.dataset == "cast20":
                    parts = batch_sample_id[i].split('-')[1].split('_')
                    if len(parts) >= 2:
                        sample_id = '_'.join(parts[1:])
                    else:
                        sample_id = batch_sample_id[i]
                else:
                    sample_id = batch_sample_id[i]

                
                
                # qid2query[sample_id] = query_embs[i].reshape(1, -1)
                qid2query_inversion[sample_id] = query_embs_inversion[i].reshape(1, -1)
                # 将第 i 个查询嵌入向量重新形状为一个包含单一行和自动确定列数的二维数组

                
    torch.cuda.empty_cache()
    return qid2query_inversion

def calculate_similarity(embedding,text,tokenizer,model):
    model = model.cuda()
    embedding1 = embedding.squeeze().detach().cpu().numpy()
    input_ids2 = tokenizer.encode(text,add_special_tokens = True,max_length = 512)
    input_ids2,attentiona_mask2 = padding_seq_to_same_length(input_ids2,max_pad_length = 512)
    input_ids2 = torch.tensor(input_ids2,dtype=torch.long).unsqueeze(0).cuda()
    attentiona_mask2 = torch.tensor(attentiona_mask2,dtype=torch.long).unsqueeze(0).cuda()
    embedding2 = model(input_ids2,attentiona_mask2).squeeze().detach().cpu().numpy()
    sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    similarity = min(1.0, max(-1.0, sim))
    return similarity
    
    
    

def invert_embeddings_to_text(args,qid2query_inversion,qid2rewrite_input_ids=None,qid2rewrite_attention_mask=None,qid2rewrite_embedding=None):
    corrector = load_corrector(args.vec2text_model)
    
    
    logger.info("inversion,corrector and tokenizer is loaded!")
    
    #sentence = "I want to eat apple! I don't want to go to school"
    

    similarity_tokenizer,similarity_encoder = load_model(args.model_type, "query", args.rewrite_checkpoint_path)

    with open(args.test_file_path, 'r') as f:
        data = f.readlines()

    output_file = oj(args.output_path, 'record.json')
    
    
    counts = 0
    similarity_sum = 0
    with open(output_file, "w") as g: 
        bt_record = []
        bt_inversion_embedding = []
        bt_rewrite_input_ids = []
        bt_rewrite_attention_mask = []
        bt_rewrite_embedding = []


        bt_index = 0
        
        for record in data:
            counts += 1
            output_record = {}
            record = json.loads(record.strip())
            if args.dataset == "cast20":
                parts = record['sample_id'].split('-')[1].split('_')
                if len(parts) >= 2:
                    sample_id = '_'.join(parts[1:])
                else:
                    sample_id = record['sample_id']
            else:
                sample_id = record['sample_id']
            output_record['sample_id'] = sample_id
            output_record['cur_utt_text'] = record['cur_utt_text']
            output_record['ctx_utts_text'] = record['ctx_utts_text']
            bt_record.append(output_record)
           
            #print(f"sample_id: {sample_id}")
           
            #print(f"cur_utt_text: {record['cur_utt_text']}")
           
            #print(f"ctx_utts_text: {record['ctx_utts_text']}")
           
            #print(f"human rewrite: {record['oracle_utt_text']}")
            
            bt_index += 1
            
            bt_inversion_embedding.append(qid2query_inversion[sample_id])
            if args.use_human_rewrite or args.use_T5QR_rewrite:
                bt_rewrite_input_ids.append(qid2rewrite_input_ids[sample_id])
                bt_rewrite_attention_mask.append(qid2rewrite_attention_mask[sample_id])
                bt_rewrite_embedding.append(qid2rewrite_embedding[sample_id])
                
            if bt_index == args.inversion_batch_size:
                embeddings = torch.cat(bt_inversion_embedding, dim=0)
                embeddings = embeddings.to(args.device)
                if args.use_human_rewrite or args.use_T5QR_rewrite:
                    rewrite_input_ids = torch.cat(bt_rewrite_input_ids, dim=0).to(args.device)
                    rewrite_attention_mask = torch.cat(bt_rewrite_attention_mask, dim=0).to(args.device)
                    rewrite_embeddings = torch.cat(bt_rewrite_embedding, dim=0).to(args.device)
                    
                    #print(embeddings.shape)
                    #print(rewrite_input_ids)
                    #print(rewrite_embeddings.shape)
                    torch.backends.cudnn.enabled = False
                    texts = invert_embeddings(embeddings = embeddings,
                                    corrector = corrector,
                                    num_steps = args.invert_num_steps,
                                    sequence_beam_width = args.sequence_beam_width,
                                    hypothesis_input_ids= rewrite_input_ids,
                                    hypothesis_attention_mask = rewrite_attention_mask,
                                    hypothesis_embedding = rewrite_embeddings)
                      
                else:
                    texts = invert_embeddings(embeddings = embeddings,
                                    corrector = corrector,
                                    num_steps = args.invert_num_steps,
                                    sequence_beam_width = args.sequence_beam_width)
                
                for index, text in enumerate(texts):
                    bt_record[index]["inversion_text"] = text
                    json.dump(bt_record[index], g)
                    g.write('\n')
                    print(bt_record[index])
                    #print(f"inversion_text: {text}")
                    sim_inversion = calculate_similarity(embeddings[index],text,similarity_tokenizer,similarity_encoder)
                    print(f"similarity_inversion: {sim_inversion}")  
                    similarity_sum += sim_inversion
                     
                bt_record = []
                bt_index = 0
                bt_rewrite_input_ids = []
                bt_rewrite_attention_mask = []
                bt_rewrite_embedding = []
                bt_inversion_embedding = []
        
        if bt_inversion_embedding != []:
            embeddings = torch.cat(bt_inversion_embedding, dim=0)
            embeddings = embeddings.to(args.device)
            if args.use_human_rewrite or args.use_T5QR_rewrite:
                rewrite_input_ids = torch.cat(bt_rewrite_input_ids, dim=0).to(args.device)
                rewrite_attention_mask = torch.cat(bt_rewrite_attention_mask, dim=0).to(args.device)
                rewrite_embeddings = torch.cat(bt_rewrite_embedding, dim=0).to(args.device)
                texts = invert_embeddings(embeddings = embeddings,
                                    corrector = corrector,
                                    num_steps = args.invert_num_steps,
                                    sequence_beam_width = args.sequence_beam_width,
                                    hypothesis_input_ids= rewrite_input_ids,
                                    hypothesis_attention_mask = rewrite_attention_mask,
                                    hypothesis_embedding = rewrite_embeddings)
                        
            else:
                texts = invert_embeddings(embeddings = embeddings,
                                        corrector = corrector,
                                        num_steps = args.invert_num_steps,
                                        sequence_beam_width = args.sequence_beam_width)
                    
            for index, text in enumerate(texts):
                bt_record[index]["inversion_text"] = text
                json.dump(bt_record[index], g)
                g.write('\n')
                print(bt_record[index])
                #print(f"inversion_text: {text}")
                sim_inversion = calculate_similarity(embeddings[index],text,similarity_tokenizer,similarity_encoder)
                print(f"similarity_inversion: {sim_inversion}")  
                similarity_sum += sim_inversion
    
    avarage_similarity = similarity_sum/counts    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"avarage_similarity_inversion: {avarage_similarity}")    
                
                
            
           
    
    
def get_args():
    parser = argparse.ArgumentParser()
    
    # test dataset
    parser.add_argument("--dataset", type=str, required=True, choices=["cast19", "cast20", "cast21", "qrecc", "topiocqa"])

    # test model
    parser.add_argument("--model_type", type=str, required=True, choices=["ance", "dpr-nq", "tctcolbert","gtr-base","bge"])
    parser.add_argument("--vec2text_model", type=str, required=True, choices=["text-embedding-ance", "text-embedding-gtr_st","text-embedding-bge"])
    parser.add_argument("--input_type", type=str, required=True, choices=["flat_concat", "raw", "oracle","concat_for_inversion","concat_for_inversion_for_conv"])
    parser.add_argument("--model_checkpoint_path", type=str, required = True, help="The tested conversational query encoder path.")
    parser.add_argument("--rewrite_checkpoint_path", type=str, required = True, help="The rewrite query encoder path.")
    parser.add_argument("--collate_fn_type", type=str, required=True, choices=["flat_concat_for_train", "flat_concat_for_test"], help="To control how to organize the batch data. Same as in the train_model.py")
    parser.add_argument("--max_query_length", type=int, default=48, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, required=True, help="Max response length, 64 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, required=True, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--enable_last_response", action="store_true", help="True for CAsT-20")

    # test input file
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--T5QR_rewrite_file_path", type=str)
    parser.add_argument("--use_human_rewrite", action="store_true", help="whether use human rewrite")
    parser.add_argument("--use_T5QR_rewrite", action="store_true", help="whether use T5QR rewrite")
    
    # test parameters 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--inversion_batch_size", type=int, default=4)
    parser.add_argument("--use_data_percent", type=float, default=1.0)
    parser.add_argument("--use_gpu_to_get_query_embedding", action="store_true", help="whether to use gpu to get query embeddings.")
    
    # vec2text parameters
    parser.add_argument("--invert_num_steps", type=int, default=20)
    parser.add_argument("--sequence_beam_width", type=int, default=5)

    # output file
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")


    # main
    args = parser.parse_args()
    device = torch.device("cuda" if args.use_gpu_to_get_query_embedding and torch.cuda.is_available() else "cpu")
    #device = "cpu"
    args.device = device

    check_dir_exist_or_build([args.output_path], args.force_emptying_dir)
    json_dumps_arguments(oj(args.output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)
    
    return args



if __name__ == '__main__':
    args = get_args()
    set_seed(args) 
    qid2query_inversion = get_query_embedding(args)   
    logger.info("query embedding is done!")
    if args.use_T5QR_rewrite or args.use_human_rewrite:
        qid2rewrite_input_ids,qid2rewrite_attention_mask,qid2rewrite_embedding = get_rewrite_embedding(args)
        logger.info("rewrite embedding is done!")
        invert_embeddings_to_text(args, qid2query_inversion,qid2rewrite_input_ids,qid2rewrite_attention_mask,qid2rewrite_embedding)
    else:
        invert_embeddings_to_text(args, qid2query_inversion)
    logger.info("invert to text has been finished!")
    