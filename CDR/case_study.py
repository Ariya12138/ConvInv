from http.cookiejar import LoadError
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/evaluation')
import os
import json
import faiss
import time
import copy
import pickle
import argparse
import numpy as np
import pytrec_eval
from tqdm import tqdm
from pprint import pprint
from os.path import join as oj

import torch
from torch.utils.data import DataLoader

from models_conv import load_model
from utils_conv import check_dir_exist_or_build, json_dumps_arguments, set_seed, load_collection, get_has_gold_label_test_qid_set, eval_run_with_qrel
from convsearch_dataset import QReCCDataset, CAsTDataset, TopiOCQADataset

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
1. get args
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''



def build_faiss_index(args):
    logger.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu_for_faiss
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(args.embedding_size)  
    index = None
    if args.use_gpu_in_faiss:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index



def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    merged_candidate_matrix = None
    if args.passage_block_num < 0:
        # automaticall get the number of passage blocks
        for filename in os.listdir(passge_embeddings_dir):
            try:
                args.passage_block_num = max(args.passage_block_num, int(filename.split(".")[1]) + 1)
            except:
                continue
        print("Automatically detect that the number of doc blocks is: {}".format(args.passage_block_num))
    for block_id in range(args.passage_block_num):
        logger.info("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(oj(passge_embeddings_dir, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(oj(passge_embeddings_dir, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
                passage_embedding2id = pickle.load(handle)
                if isinstance(passage_embedding2id, list):
                    passage_embedding2id = np.array(passage_embedding2id)
        except:
            raise LoadError    
        
        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))

        passage_embeddings = np.array_split(passage_embedding, args.num_split_block)
        passage_embedding2ids = np.array_split(passage_embedding2id, args.num_split_block)
        for split_idx in range(len(passage_embeddings)):
            passage_embedding = passage_embeddings[split_idx]
            passage_embedding2id = passage_embedding2ids[split_idx]
            
            logger.info("Adding block {} split {} into index...".format(block_id, split_idx))
            index.add(passage_embedding)
            
            # ann search
            tb = time.time()
            D, I = index.search(query_embeddings, topN)
            elapse = time.time() - tb
            logger.info({
                'time cost': elapse,
                'query num': query_embeddings.shape[0],
                'time cost per query': elapse / query_embeddings.shape[0]
            })

            candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
            D = D.tolist()
            candidate_id_matrix = candidate_id_matrix.tolist()
            candidate_matrix = []

            for score_list, passage_list in zip(D, candidate_id_matrix):
                candidate_matrix.append([])
                for score, passage in zip(score_list, passage_list):
                    candidate_matrix[-1].append((score, passage))
                assert len(candidate_matrix[-1]) == len(passage_list)
            assert len(candidate_matrix) == I.shape[0]

            index.reset()
            del passage_embedding
            del passage_embedding2id

            if merged_candidate_matrix == None:
                merged_candidate_matrix = candidate_matrix
                continue
            
            # merge
            merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
            merged_candidate_matrix = []
            for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
                p1, p2 = 0, 0
                merged_candidate_matrix.append([])
                while p1 < topN and p2 < topN:
                    if merged_list[p1][0] >= cur_list[p2][0]:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    else:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1
                while p1 < topN:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                while p2 < topN:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1

    merged_D, merged_I = [], []

    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    logger.info(merged_D.shape)
    logger.info(merged_I.shape)

    return merged_D, merged_I


def output_test_result(args, 
                       queryembedding2sampleid,
                       retrieved_scores_mat,
                       retrieved_pid_mat):
    
    qids_to_ranked_candidate_passages = {}
    topN = args.top_n

    # 用于构建查询的排名列表qids_to_ranked_candidate_passages，
    # 确保列表中的文档不重复，并在达到 topN 时停止。
    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = queryembedding2sampleid[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            # 这部分检查查询是否已经有了排名列表。如果已经有了，就跳过；
            # 否则，为查询创建一个长度为 topN 的初始排名列表。
            tmp = [(0, 0)] * topN
            # tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for pred_pid, score in zip(selected_ann_idx, selected_ann_score):
            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
    # query
    qid2query = {}
    with open(args.test_file_path, 'r') as f:
        data = f.readlines()
    for record in data:
        record = json.loads(record.strip())

        if args.dataset == "cast20" and args.test_input_type != "inversion_text":
            parts = record['sample_id'].split('-')[1].split('_')
            if len(parts) >= 2:
                sample_id = '_'.join(parts[1:])
            else:
                sample_id = record['sample_id']
        else:
            sample_id = record['sample_id']
        qid2query[sample_id] = record['cur_utt_text']

    
    # all passages
    if args.output_passage_content:
        all_passages = load_collection(args.passage_collection_path)
    
    # write to file
    logger.info('begin to write the output...')

    output_file = oj(args.retrieval_output_path, 'run.json')
    output_trec_file = oj(args.retrieval_output_path, 'run.trec')
    with open(output_file, "w") as f, open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            query = qid2query[qid]
            for i in range(topN):
                pid, score = passages[i]
                passage = ""
                if args.output_passage_content:
                    passage = all_passages[pid]

                f.write(
                        json.dumps({
                            "sample_id": str(qid),
                            "cur_utt_text": query,
                            "doc": passage,
                            "doc_id": str(pid),
                            "rank": i,
                            "retrieval_score": score,
                        }) + "\n")
                
                g.write(
                        str(qid) + " Q0 " + str(pid) + " " + str(i + 1) +
                        " " + str(-i + 1000) + " gtr\n")
    
    logger.info("output file write ok at {}".format(args.retrieval_output_path))

    # evaluate
    eval_kwargs = {"run_file": output_trec_file, 
                   "qrel_file": args.gold_qrel_file_path, 
                   "rel_threshold": args.rel_threshold,
                   "retrieval_output_path": args.retrieval_output_path}
    res = eval_run_with_qrel(**eval_kwargs)

    return res


def faiss_retrieval(args, index, query_embeddings, queryembedding2sampleid):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                     args,
                                                     args.doc_embeddings_dir_path, 
                                                     index, 
                                                     query_embeddings, 
                                                     args.top_n) 

    output_test_result(args, queryembedding2sampleid, retrieved_scores_mat, retrieved_pid_mat)



def get_args():
    parser = argparse.ArgumentParser()
    
    # test dataset
    parser.add_argument("--dataset", type=str, required=True, choices=["cast19", "cast20", "cast21", "qrecc", "topiocqa"])

    # test model
    parser.add_argument("--model_type", type=str, required=True, choices=["ance", "dpr-nq", "tctcolbert","gtr-base"])
    parser.add_argument("--test_input_type", type=str, required=True, choices=["flat_concat", "raw", "oracle","concat_for_inversion","inversion_text"])
    parser.add_argument("--model_checkpoint_path", type=str, required = True, help="The tested conversational query encoder path.")
    
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, required=True, help="Max response length, 64 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, required=True, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--enable_last_response", action="store_true", help="True for CAsT-20")

    # test input file
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--doc_embeddings_dir_path", type=str, required=True)
    parser.add_argument("--passage_block_num", type=int, default=-1, help="As the passage embeddings are too large \
                                                                            to be loaded in the memory at one time, \
                                                                            we split docs into several blocks. \
                                                                            setting -1 is to automatically get the num block.")
    parser.add_argument("--gold_qrel_file_path", type=str, required=True)
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")
    
    # test parameters 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--use_gpu_in_faiss", action="store_true", help="whether to use gpu in faiss or not.")
    parser.add_argument("--n_gpu_for_faiss", type=int, default=1, help="should be set if use_gpu_in_faiss")
    parser.add_argument("--use_gpu_to_get_query_embedding", action="store_true", help="whether to use gpu to get query embeddings.")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--use_data_percent", type=float, default=1.0)
    parser.add_argument("--num_split_block", type=int, default=1, help="further split each block into several sub-blocks to reduce gpu memory use.")
    parser.add_argument("--output_passage_content", action="store_true", help="need to output passage content or not.")
    parser.add_argument("--passage_collection_path", type=str, default="", help="need to be set if output_passage_content is True.")
    
    # output file
    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")


    # main
    args = parser.parse_args()
    device = torch.device("cuda" if args.use_gpu_to_get_query_embedding and torch.cuda.is_available() else "cpu")
    args.device = device

    check_dir_exist_or_build([args.retrieval_output_path], args.force_emptying_dir)
    json_dumps_arguments(oj(args.retrieval_output_path, "parameters.txt"), args)
 
    logger.info("---------------------The arguments are:---------------------")
    pprint(args)
    
    return args





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


def get_test_query_embedding(args,sample_id,inversion_text,is_session = False):
    query_tokenizer, query_encoder = load_model(args.model_type, "query", args.model_checkpoint_path)
    query_encoder = query_encoder.to(args.device)
    if is_session == True:
        print(inversion_text)
        inversion_utt = query_tokenizer.encode(inversion_text, add_special_tokens = True, max_length = args.max_concat_length)
        inversion, mask_inversion = padding_seq_to_same_length(inversion_utt, max_pad_length = args.max_concat_length)
    else:
        inversion_utt = query_tokenizer.encode(inversion_text, add_special_tokens = True, max_length = args.max_query_length)
        inversion, mask_inversion = padding_seq_to_same_length(inversion_utt, max_pad_length = args.max_query_length)

    inversion = torch.tensor(inversion).unsqueeze(0).to(args.device)
    mask_inversion = torch.tensor(mask_inversion).unsqueeze(0).to(args.device)
    
    query_embs = query_encoder(inversion, mask_inversion)
    query_embeddings = query_embs.detach().cpu().numpy()
    queryembedding2sampleid = [sample_id]

    print(query_embeddings.shape)
    print(queryembedding2sampleid)
    return query_embeddings,queryembedding2sampleid




if __name__ == '__main__':
    args = get_args()
    set_seed(args) 
    
    index = build_faiss_index(args)
    inversion_text = "What is the functionalist theory?How is his work related to Comte?What is Herbert Spencer known for?What is the role of positivism in it?What is the main contribution of Auguste Comte?What is taught in sociology?"    
    #inversion_text = "Tell me about the author of the experiment."
    #inversion_text += "What did it show?"
    #inversion_text += "What was the Stanford Experiment?"
    
    query_embeddings, queryembedding2sampleid = get_test_query_embedding(args,"79_6",inversion_text,True)
    faiss_retrieval(args, index, query_embeddings, queryembedding2sampleid)

    logger.info("Test finish!")
    
