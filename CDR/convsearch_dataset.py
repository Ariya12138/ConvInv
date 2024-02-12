#from IPython import embed
import sys
sys.path.append('..')
sys.path.append('.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import json
import random
from tqdm import trange, tqdm
from abc import abstractmethod

import torch
from torch.utils.data import Dataset


# "_text" means the raw text data
class ConversationalSearchSample:
    def __init__(self, sample_id, 
                       cur_utt_text,
                       oracle_utt_text,
                       ctx_utts_text,
                       last_response_text,
                       inversion_utt,
                       cur_utt = None,  # model input
                       oracle_utt = None, # model input
                       flat_concat = None,   # model_input
                       pos_docs = None,
                       neg_docs = None,
                       cur_utt_end_position = None):
        self.sample_id = sample_id
        # original text info
        self.cur_utt_text = cur_utt_text # current utterance 当前查询
        self.oracle_utt_text = oracle_utt_text # oracle utterance text 手写的查询重写
        self.ctx_utts_text = ctx_utts_text # 对话历史
        self.last_response_text = last_response_text    # mainly for CAsT-20
        self.inversion_utt = inversion_utt
        

        # tokenized model input
        self.cur_utt = cur_utt # embedding of current utterance 当前查询的embedding
        self.oracle_utt = oracle_utt # embedding of oracle utt 手写查询重写的embedding
        self.flat_concat = flat_concat
        
        
        # docs (model input)
        self.pos_docs = pos_docs # positive docs embedding
        self.neg_docs = neg_docs # negative docs embedding

        self.cur_utt_end_position = None    # the end position of the last token of cur_utt


class ConversationalSearchDataset(Dataset):
    def __init__(self, args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info=True):
        self.examples = []
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

        self.max_query_length = args.max_query_length
        self.max_doc_length = args.max_doc_length
        self.max_response_length = args.max_response_length
        self.max_concat_length = args.max_concat_length
        self.enable_last_response = args.enable_last_response

        self.need_doc_info = need_doc_info
        if self.need_doc_info:
            self.negative_type = args.negative_type
            self.neg_ratio = args.neg_ratio

        with open(data_filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for debugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Processing {} data file...".format(data_filename))
        for i in trange(n):
            data[i] = json.loads(data[i])
            sample_id = data[i]['sample_id']
            
            cur_utt_text = data[i]['cur_utt_text']
            if "oracle_utt_text" in data[i]:  
                oracle_utt_text = data[i]['oracle_utt_text']
            else:
                oracle_utt_text = ""
            ctx_utts_text = data[i]['ctx_utts_text']
            last_response_text = None
            if "last_response_text" in data[i]:
                last_response_text = data[i]['last_response_text']

            inversion_utt = None
            if "inversion_text" in data[i]:
                inversion_text = data[i]['inversion_text']
                inversion_utt = query_tokenizer.encode(inversion_text, add_special_tokens = True, max_length = args.max_query_length)

                
            if args.model_type == "TCT-ColBERT":
                prefix = "[ Q ] "
                cur_utt_text = prefix + cur_utt_text
                oracle_utt_text = prefix + oracle_utt_text

            
            cur_utt = query_tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            oracle_utt = query_tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)

            example = ConversationalSearchSample(sample_id, 
                                                 cur_utt_text, 
                                                 oracle_utt_text, 
                                                 ctx_utts_text, 
                                                 last_response_text,
                                                 inversion_utt, 
                                                 cur_utt, 
                                                 oracle_utt)
            
            if self.need_doc_info:
                have_pos_doc, example = self.add_pos_and_neg_doc_info(example, data[i])
                if have_pos_doc:
                    self.examples.append(example)
            else:
                self.examples.append(example)
 
    # 如果数据集中原本就有对应的正例positive examples才能进行训练
    # 对负例negative examples进行选择后训练。负例来源分为random_neg，bm25_hard_neg与in_batch_neg。
    def add_pos_and_neg_doc_info(self, example, raw_data):
        if len(raw_data['pos_docs_pids']) == 0:
            return False, example

        # doc info for ranking loss
        pos_docs = []
                
        # pos_docs
        for doc in raw_data['pos_docs_text']:
            pos_docs.append(self.doc_tokenizer.encode(doc, add_special_tokens=True, max_length=self.max_doc_length))
        example.pos_docs = pos_docs
        
        # neg docs
        if self.negative_type == "random_neg":
            assert "random_neg_docs_text" in raw_data
            random_neg_docs = []
            for doc in raw_data['random_neg_docs_text']:
                random_neg_docs.append(self.doc_tokenizer.encode(doc, add_special_tokens=True, max_length=self.max_doc_length))
                if len(random_neg_docs) == self.neg_ratio:
                    break
            example.neg_docs = random_neg_docs
        elif self.negative_type == "bm25_hard_neg":
            assert "bm25_hard_neg_docs_text" in raw_data
            bm25hard_neg_docs = []
            for doc in raw_data['bm25_hard_neg_docs_text']:
                bm25hard_neg_docs.append(self.doc_tokenizer.encode(doc, add_special_tokens=True, max_length=self.max_doc_length))
                if len(bm25hard_neg_docs) == self.neg_ratio:
                    break
            example.neg_docs = bm25hard_neg_docs
        elif self.negative_type == "in_batch_neg":
            example.neg_docs = None
        else:
            raise KeyError("Negative type: {} not implmeneted".format(self.negative_type))
    
        return True, example 

    def add_flat_concat_for_inversion_to_example(self):
        for i in range(len(self.examples)):
            ctx_utts_text = self.examples[i].ctx_utts_text
            last_response_text = self.examples[i].last_response_text

            full_text = self.examples[i].cur_utt_text
           
            if self.enable_last_response and len(last_response_text) > 0:
                # still use query tokenizer!
                full_text += last_response_text

            for j in range(len(ctx_utts_text) - 1, -1, -1):
                full_text += ctx_utts_text[j]

            #print(full_text)
            flat_concat_inversion = self.query_tokenizer.encode(full_text, add_special_tokens=True,max_length=self.max_concat_length) # not remove [CLS]
            self.examples[i].flat_concat_inversion = flat_concat_inversion
            

        return
    
    def add_flat_concat_for_convInversion_to_example(self):
        for i in range(len(self.examples)):
            ctx_utts_text = self.examples[i].ctx_utts_text
            last_response_text = self.examples[i].last_response_text

            to_concat = []
            to_concat.extend(ctx_utts_text)
            
            sep_token = self.query_tokenizer.sep_token
            if sep_token is None:
                sep_token = self.query_tokenizer.eos_token

            if self.enable_last_response and len(last_response_text) > 0:
                # still use query tokenizer!
                to_concat.append(last_response_text)
                
            to_concat.append(self.examples[i].cur_utt_text)
            to_concat.reverse()

            text = " {} ".format(sep_token).join(to_concat)

            flat_concat_inversion = self.query_tokenizer.encode(text, max_length=self.max_concat_length) # not remove [CLS]
            self.examples[i].flat_concat_inversion = flat_concat_inversion
            
        return
    
    def add_flat_concat_with_blank_space(self):
        for i in range(len(self.examples)):
            ctx_utts_text = self.examples[i].ctx_utts_text
            last_response_text = self.examples[i].last_response_text

            to_concat = []
            to_concat.extend(ctx_utts_text)
            
            sep_token = " "

            if self.enable_last_response and len(last_response_text) > 0:
                # still use query tokenizer!
                to_concat.append(last_response_text)
                
            to_concat.append(self.examples[i].cur_utt_text)
            to_concat.reverse()

            text = "{}".format(sep_token).join(to_concat)
            print(text)
            flat_concat_inversion = self.query_tokenizer.encode(text, add_special_tokens=True,max_length=self.max_concat_length) # not remove [CLS]
            self.examples[i].flat_concat_inversion = flat_concat_inversion
            
        return
    
    # 用于定义抽象方法，要求子类必须实现这些方法。
    @abstractmethod
    def add_flat_concat_to_example(self):
        print("Add \"flat_concat\" variable to all examples.")
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
        

    
    

    # 用于定义静态方法，这些方法与类相关，但不涉及实例或实例属性。
    @staticmethod
    def get_collate_fn(args):
        # collate_fn: for "flat concat for train"
        def collate_fn_flat_concat_for_train(batch):
            res_batch_info = {
                "bt_sample_id": [],
                "bt_concat":[],
                "bt_concat_mask":[],
                "bt_oracle_utt":[],
                "bt_oracle_utt_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
                "bt_cur_utt_end_positions":[],
                "bt_concat_inversion":[],
                "bt_concat_mask_inversion":[]
            }

            bt_sample_id = [] 
            bt_concat = []
            bt_concat_mask = []
            bt_oracle_utt = []
            bt_oracle_utt_mask = []
            bt_cur_utt_end_positions = []
            
            # docs, for training with ranking loss.
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []

            bt_concat_inversion = []
            bt_concat_mask_inversion = []
            
            for example in batch:
                # padding
                concat, concat_mask = padding_seq_to_same_length(example.flat_concat, max_pad_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                bt_concat.append(concat)
                bt_concat_mask.append(concat_mask)
                bt_cur_utt_end_positions.append(example.cur_utt_end_position)
                
                oracle_utt, oracle_utt_mask = padding_seq_to_same_length(example.oracle_utt, max_pad_length = args.max_query_length)
                bt_oracle_utt.append(oracle_utt)
                bt_oracle_utt_mask.append(oracle_utt_mask)

                concat_inversion, concat_mask_inversion = padding_seq_to_same_length(example.flat_concat_inversion, max_pad_length = args.max_concat_length)
                bt_concat_inversion.append(concat_inversion)
                bt_concat_mask_inversion.append(concat_mask_inversion)
                
                if args.need_doc_info:
                    # Ranking loss
                    # 1 pos_doc
                    pos_doc = random.sample(example.pos_docs, 1)[0]
                    pos_doc, pos_doc_mask = padding_seq_to_same_length(pos_doc, max_pad_length = args.max_doc_length)
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)

                    # neg_ratio neg_docs
                    # 一个emaple可能有多个negative doc
                    neg_docs = example.neg_docs
                    tmp_neg_docs, tmp_neg_docs_mask = [], []    # one example has multiple neg docs
                    if neg_docs is None:
                        continue
                    for doc in neg_docs:
                        doc, doc_mask = padding_seq_to_same_length(doc, max_pad_length = args.max_doc_length)
                        tmp_neg_docs.append(doc)
                        tmp_neg_docs_mask.append(doc_mask)
                    bt_neg_docs.append(tmp_neg_docs)
                    bt_neg_docs_mask.append(tmp_neg_docs_mask)

            res_batch_info["bt_sample_id"] = bt_sample_id
            res_batch_info["bt_concat"] = bt_concat
            res_batch_info["bt_concat_mask"] = bt_concat_mask
            res_batch_info["bt_concat_inversion"] = bt_concat_inversion
            res_batch_info["bt_concat_mask_inversion"] = bt_concat_mask_inversion
            res_batch_info["bt_oracle_utt"] = bt_oracle_utt
            res_batch_info["bt_oracle_utt_mask"] = bt_oracle_utt_mask
            res_batch_info["bt_cur_utt_end_positions"] = bt_cur_utt_end_positions
            res_batch_info["bt_pos_docs"] = bt_pos_docs
            res_batch_info["bt_pos_docs_mask"] = bt_pos_docs_mask
            res_batch_info["bt_neg_docs"] = bt_neg_docs
            res_batch_info["bt_neg_docs_mask"] = bt_neg_docs_mask

            # change to tensor
            for key in res_batch_info:
                if key not in ["bt_sample_id"]:
                    res_batch_info[key] = torch.tensor(res_batch_info[key], dtype=torch.long)
                    
            return res_batch_info

        # collate_fn: for "flat concat for test"
        def collate_fn_flat_concat_for_test(batch):
            res_batch_info = {
                "bt_sample_id": [],
                "bt_concat":[],
                "bt_concat_mask":[],
                "bt_oracle_utt":[],
                "bt_oracle_utt_mask":[],
                "bt_cur_utt":[],
                "bt_cur_utt_mask":[],
                "bt_concat_inversion":[],
                "bt_concat_mask_inversion":[],
                #"bt_concat_inversion_conv":[],
                #"bt_concat_mask_inversion_conv":[],
                "bt_inversion":[],
                "bt_mask_inversion":[]
            }

            bt_sample_id = [] 
            bt_concat = []
            bt_concat_mask = []
            bt_oracle_utt = []
            bt_oracle_utt_mask = []
            bt_cur_utt = []
            bt_cur_utt_mask = []
            bt_cur_utt_end_positions = []
            bt_concat_inversion = []
            bt_concat_mask_inversion = []
            #bt_concat_inversion_conv = []
            #bt_concat_mask_inversion_conv = []
            bt_inversion = []
            bt_mask_inversion = []
            
        
            for example in batch:
                bt_sample_id.append(example.sample_id)

                # padding
                concat, concat_mask = padding_seq_to_same_length(example.flat_concat, max_pad_length = args.max_concat_length)
                bt_concat.append(concat)
                bt_concat_mask.append(concat_mask)
                
                oracle_utt, oracle_utt_mask = padding_seq_to_same_length(example.oracle_utt, max_pad_length = args.max_query_length)
                bt_oracle_utt.append(oracle_utt)
                bt_oracle_utt_mask.append(oracle_utt_mask)
                
                cur_utt, cur_utt_mask = padding_seq_to_same_length(example.cur_utt, max_pad_length = args.max_query_length)
                bt_cur_utt.append(cur_utt)
                bt_cur_utt_mask.append(cur_utt_mask)

                bt_cur_utt_end_positions.append(example.cur_utt_end_position)

                concat_inversion, concat_mask_inversion = padding_seq_to_same_length(example.flat_concat_inversion, max_pad_length = args.max_concat_length)
                bt_concat_inversion.append(concat_inversion)
                bt_concat_mask_inversion.append(concat_mask_inversion)

                '''
                concat_inversion_conv, concat_mask_inversion_conv = padding_seq_to_same_length(example.flat_concat_inversion_conv, max_pad_length = args.max_concat_length)
                bt_concat_inversion_conv.append(concat_inversion_conv)
                bt_concat_mask_inversion_conv.append(concat_mask_inversion_conv)
                '''

                if example.inversion_utt != None:
                    inversion, mask_inversion = padding_seq_to_same_length(example.inversion_utt, max_pad_length = args.max_query_length)
                    bt_inversion.append(inversion)
                    bt_mask_inversion.append(mask_inversion)

                

            res_batch_info["bt_sample_id"] = bt_sample_id
            res_batch_info["bt_concat"] = bt_concat
            res_batch_info["bt_concat_mask"] = bt_concat_mask
            res_batch_info["bt_oracle_utt"] = bt_oracle_utt
            res_batch_info["bt_oracle_utt_mask"] = bt_oracle_utt_mask
            res_batch_info["bt_cur_utt"] = bt_cur_utt
            res_batch_info["bt_cur_utt_mask"] = bt_cur_utt_mask
            res_batch_info["bt_cur_utt_end_positions"] = bt_cur_utt_end_positions
            res_batch_info["bt_concat_inversion"] = bt_concat_inversion
            res_batch_info["bt_concat_mask_inversion"] = bt_concat_mask_inversion
            #res_batch_info["bt_concat_inversion_conv"] = bt_concat_inversion_conv
            #res_batch_info["bt_concat_mask_inversion_conv"] = bt_concat_mask_inversion_conv
            res_batch_info["bt_inversion"] = bt_inversion
            res_batch_info["bt_mask_inversion"] = bt_mask_inversion
            

            # change to tensor
            for key in res_batch_info:
                if key not in ["bt_sample_id"] and res_batch_info[key] != None:
                    res_batch_info[key] = torch.tensor(res_batch_info[key], dtype=torch.long)
                    
            return res_batch_info
        # collate_fn: for "flat concat for test"
        
        def collate_fn_flat_concat_for_test_with_text_info(batch):
            res_batch_info = {
                "bt_sample_id": [],
                "bt_cur_utt_text": [],
                "bt_oracle_utt_text": [],
                "bt_ctx_utts_text": [],
                "bt_concat":[],
                "bt_concat_mask":[],
                "bt_oracle_utt":[],
                "bt_oracle_utt_mask":[],
                "bt_cur_utt":[],
                "bt_cur_utt_mask":[],
            }

            bt_sample_id = [] 
            bt_concat = []
            bt_concat_mask = []
            bt_oracle_utt = []
            bt_oracle_utt_mask = []
            bt_cur_utt = []
            bt_cur_utt_mask = []
            bt_cur_utt_end_positions = []
        
            bt_cur_utt_text = []
            bt_oracle_utt_text = []
            bt_ctx_utts_text = []

            for example in batch:
                bt_sample_id.append(example.sample_id)

                # padding
                concat, concat_mask = padding_seq_to_same_length(example.flat_concat, max_pad_length = args.max_concat_length)
                bt_concat.append(concat)
                bt_concat_mask.append(concat_mask)
                
                oracle_utt, oracle_utt_mask = padding_seq_to_same_length(example.oracle_utt, max_pad_length = args.max_query_length)
                bt_oracle_utt.append(oracle_utt)
                bt_oracle_utt_mask.append(oracle_utt_mask)
                
                cur_utt, cur_utt_mask = padding_seq_to_same_length(example.cur_utt, max_pad_length = args.max_query_length)
                bt_cur_utt.append(cur_utt)
                bt_cur_utt_mask.append(cur_utt_mask)

                bt_cur_utt_end_positions.append(example.cur_utt_end_position)

                bt_cur_utt_text.append(example.cur_utt_text)
                bt_oracle_utt_text.append(example.oracle_utt_text)
                bt_ctx_utts_text.append(example.ctx_utts_text)




            res_batch_info["bt_sample_id"] = bt_sample_id
            res_batch_info["bt_concat"] = bt_concat
            res_batch_info["bt_concat_mask"] = bt_concat_mask
            res_batch_info["bt_oracle_utt"] = bt_oracle_utt
            res_batch_info["bt_oracle_utt_mask"] = bt_oracle_utt_mask
            res_batch_info["bt_cur_utt"] = bt_cur_utt
            res_batch_info["bt_cur_utt_mask"] = bt_cur_utt_mask
            res_batch_info["bt_cur_utt_end_positions"] = bt_cur_utt_end_positions

            res_batch_info["bt_cur_utt_text"] = bt_cur_utt_text
            res_batch_info["bt_oracle_utt_text"] = bt_oracle_utt_text
            res_batch_info["bt_ctx_utts_text"] = bt_ctx_utts_text

            # change to tensor
            for key in res_batch_info:
                if key not in ["bt_sample_id", "bt_cur_utt_text", "bt_oracle_utt_text", "bt_ctx_utts_text"]:
                    res_batch_info[key] = torch.tensor(res_batch_info[key], dtype=torch.long)
                    
            return res_batch_info
        
            
        

        CollateFnTypes = {
            "flat_concat_for_train": collate_fn_flat_concat_for_train,
            "flat_concat_for_test": collate_fn_flat_concat_for_test,
            "flat_concat_for_test_with_text_info": collate_fn_flat_concat_for_test_with_text_info
        }

        return CollateFnTypes[args.collate_fn_type]
    


class QReCCDataset(ConversationalSearchDataset):
    def __init__(self, args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info=True):
        super().__init__(args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info)
        self.add_flat_concat_to_example()
        self.add_flat_concat_for_inversion_to_example()
        #self.add_flat_concat_for_convInversion_to_example()
        #self.add_flat_concat_with_blank_space()

    # 将当前查询与对话历史的embedding拼接起来得到flat_concat，并记录拼接后最后一个token的位置
    def add_flat_concat_to_example(self):
        for i in range(len(self.examples)):
            flat_concat = []
            ctx_utts_text = self.examples[i].ctx_utts_text
            
            # 展开为一个单层的列表，以便所有元素都在同一层级上，而不再存在嵌套关系。
            flat_concat.extend(self.examples[i].cur_utt)
            cur_utt_end_position = len(flat_concat) - 1

            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = self.max_response_length
                else:
                    max_length = self.max_query_length
                utt = self.query_tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > self.max_concat_length:
                    flat_concat += utt[:self.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)

            self.examples[i].flat_concat = flat_concat
            self.examples[i].cur_utt_end_position = cur_utt_end_position
            
        return 
    
    


class CAsTDataset(ConversationalSearchDataset):
    def __init__(self, args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info=True):
        super().__init__(args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info)
        self.add_flat_concat_to_example()
        self.add_flat_concat_for_inversion_to_example()
        #self.add_flat_concat_for_convInversion_to_example()
        #self.add_flat_concat_with_blank_space()

    def add_flat_concat_to_example(self):
        for i in range(len(self.examples)):
            flat_concat = []
            ctx_utts_text = self.examples[i].ctx_utts_text
            last_response_text = self.examples[i].last_response_text

            flat_concat.extend(self.examples[i].cur_utt)
            cur_utt_end_position = len(flat_concat) - 1

            if self.enable_last_response and len(last_response_text) > 0:
                # still use query tokenizer!
                last_response = self.query_tokenizer.encode(last_response_text, add_special_tokens=True, max_length=self.max_response_length)
                flat_concat.extend(last_response)

            for j in range(len(ctx_utts_text) - 1, -1, -1):
                max_length = self.max_query_length
                utt = self.query_tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length) # not remove [CLS]
                if len(flat_concat) + len(utt) > self.max_concat_length:
                    flat_concat += utt[:self.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)
             
            self.examples[i].flat_concat = flat_concat
            self.examples[i].cur_utt_end_position = cur_utt_end_position

        return 
    


    
    


class TopiOCQADataset(ConversationalSearchDataset):
    def __init__(self, args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info=True):
        super().__init__(args, query_tokenizer, doc_tokenizer, data_filename, need_doc_info)
        self.add_flat_concat_to_example()

    # Same as QReCC
    def add_flat_concat_to_example(self):
        for i in range(len(self.examples)):
            flat_concat = []
            ctx_utts_text = self.examples[i].ctx_utts_text
            
            flat_concat.extend(self.examples[i].cur_utt)
            cur_utt_end_position = len(flat_concat) - 1

            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = self.max_response_length
                else:
                    max_length = self.max_query_length
                utt = self.query_tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length) # not remove [CLS]
                if len(flat_concat) + len(utt) > self.max_concat_length:
                    flat_concat += utt[:self.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)

            self.examples[i].flat_concat = flat_concat
            self.examples[i].cur_utt_end_position = cur_utt_end_position
  
        return 


class T5RewriterDataset(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True)[1:]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt)

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            oracle_utt_text = record["oracle_utt_text"]
            if args.collate_fn_type == "flat_concat_for_train":
                target_seq = oracle_utt_text
                target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
            else:
                labels = []
            
            self.examples.append([record['sample_id'], 
                                  flat_concat,
                                  flat_concat_mask,
                                  labels,
                                  cur_utt_text,
                                  oracle_utt_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn



# 将每个输入序列都扩展到相同长度
# 将输入的序列 input_ids 填充或截断到指定的最大长度 max_pad_length，
# 并生成对应的注意力掩码（attention mask）。
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



class ConvExample_rewrite:
    def __init__(self, sample_id, 
                       #query,
                       rewrite):
        self.sample_id = sample_id
        #self.query = query
        self.rewrite = rewrite


class ConvDataset_rewrite(Dataset):
    def __init__(self, args, query_tokenizer):
        self.examples = []

        with open(args.test_file_path_1, 'r') as f:
            data = f.readlines()

        with open(args.test_file_path_2, 'r') as f2:
            data_2 = f2.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)

        logger.info("Loading {} data file...".format(args.test_file_path_1))
        logger.info("Loading {} data file...".format(args.test_file_path_2))

        for i in trange(n):
            data[i] = json.loads(data[i])
            if 'id' in data[i]:
                sample_id = data[i]['id']
            else:
                sample_id = data[i]['sample_id']
            if 'output' in data[i]:
                rewrite = data[i]['output']
            elif 'rewrite' in data[i]:
                rewrite = data[i]['rewrite']
            else:
                rewrite = data[i]['oracle_utt_text']
            if 'query' in data[i]:
                cur_query = data[i]['query']
            else:
                cur_query = data[i]['cur_utt_text']
            #if "answer" in data[i]:
            #    answer = data[i]["answer"]
            #else:
            #    answer = data[i]["cur_response_text"]
            #rewrite = rewrite + ' ' + answer
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                rewrite = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                rewrite = rewrite + ' ' + data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+nexq":
                data_2[i] = json.loads(data_2[i])
                rewrite = rewrite + ' ' + data_2[i]['next_q_utt_text']

            #print(rewrite)
            rewrite = query_tokenizer.encode(rewrite, add_special_tokens=True)
            #query = query_tokenizer.encode(cur_query, add_special_tokens=True)

            self.examples.append(ConvExample_rewrite(sample_id,
                                        #query,
                                        rewrite,
                                        )) 


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_rewrite":[],
                "bt_rewrite_mask":[],
            }
            
            bt_sample_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_rewrite = []
            bt_rewrite_mask = []

            for example in batch:
                # padding
                #query, query_mask = pad_seq_ids_with_mask(example.query, max_length = args.max_query_length)
                rewrite, rewrite_mask = padding_seq_to_same_length(example.rewrite, max_pad_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                #bt_query.append(query)
                #bt_query_mask.append(query_mask)  
                bt_rewrite.append(rewrite)
                bt_rewrite_mask.append(rewrite_mask)     

            collated_dict["bt_sample_id"] = bt_sample_id
            #collated_dict["bt_query"] = bt_query
            #collated_dict["bt_query_mask"] = bt_query_mask
            collated_dict["bt_rewrite"] = bt_rewrite
            collated_dict["bt_rewrite_mask"] = bt_rewrite_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn