from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('/home/yiruo_cheng/proposal/CDRvec2text/evaluation')

from models_conv import load_model
from utils_conv import check_dir_exist_or_build, set_seed, get_optimizer, json_dumps_arguments
from convsearch_dataset import QReCCDataset, CAsTDataset, TopiOCQADataset

import time
import numpy as np
import argparse
from os.path import join as oj
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tensorboardX import SummaryWriter
from transformers import get_linear_schedule_with_warmup

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler






def save_model(model_output_path, model, query_tokenizer, epoch):
    output_dir = oj(model_output_path, 'epoch-{}'.format(epoch))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Save checkpoint at {}".format(output_dir))


def cal_kd_loss(query_embs, oracle_query_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, oracle_query_embs)

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    # query_embs是 B * dim
    # pos_doc_embs也是 B * dim
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores
    if neg_doc_embs is not None:
        neg_ratio = int(neg_doc_embs.shape[0] / query_embs.shape[0])
        # 将query_embs变为 B * 1 *dim 的格式
        # 将neg_doc_embs变为 B * neg_ratio * dim的格式
        # 进行每个元素的相乘得到 B * neg_ratio * dim
        # 最后求和得到 B * neg_ratio的neg_scores得分矩阵
        # 每个元素 (i, j) 表示第 i 个查询与第 j 个负面文档之间的得分。
        neg_scores = torch.sum(query_embs.unsqueeze(1) * neg_doc_embs.view(batch_size, neg_ratio, -1), dim = -1) # B * neg_ratio
        score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + neg_ratio)  in_batch negatives + neg_ratio other negatives
    # 创建一个张量，包含从 0 到 batch_size - 1 的整数，即 [0, 1, 2, ..., batch_size - 1]。 
    label_mat = torch.arange(batch_size).to(query_embs.device)
    loss_func = nn.CrossEntropyLoss()
    # 设置温度参数
    #adjusted_score_mat = score_mat / 0.01
    loss = loss_func(score_mat, label_mat)
    return loss


def train(args):
    if not args.need_output:
        args.log_path = "./tmp"

    # 主进程通常负责一些全局的任务，比如创建日志目录。
    # 如果当前进程是排名为 0 的主进程，就创建一个 SummaryWriter 对象，
    # 该对象用于写入 TensorBoard 日志。log_dir 参数指定了 TensorBoard 日志的保存路径。
    if dist.get_rank() == 0:
        check_dir_exist_or_build([args.log_path], args.force_emptying_dir)
        log_writer = SummaryWriter(log_dir = args.log_path)

    # 如果当前进程不是排名为 0 的主进程，那么将 log_writer 设置为 None。
    # 这是因为通常只有主进程负责创建 TensorBoard 日志，其他进程只需共享这些日志而无需创建。
    else:
        log_writer = None

    # 1. Load query and doc encoders
    query_tokenizer, query_encoder = load_model(args.model_type, "query", args.pretrained_query_encoder_path)
    query_encoder.to(args.device)
    # 在训练中使用排序损失（Ranking Loss），这种损失函数通常用于排序任务，
    # 比如在信息检索中，学习将相关文档排在不相关文档之前。
    if args.loss_type == "ranking":
        doc_tokenizer, doc_encoder = load_model(args.model_type, "doc", args.pretrained_doc_encoder_path)
        doc_encoder = doc_encoder.to(args.device)

    # 使用知识蒸馏（Knowledge Distillation）的损失函数，其中模型通过学习另一个模型的知识。
    # 在这里，可能有一个查询模型（teacher model），其知识被传递给另一个文档模型（student model）。
    elif args.loss_type == "kd":
        _, oracle_query_encoder = load_model(args.model_type, "query", args.pretrained_query_encoder_path)
        oracle_query_encoder.to(args.device)
        doc_tokenizer = None
    else:
        doc_tokenizer = None
    query_encoder = DDP(query_encoder, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
    dist.barrier()

    # 2. Prepare training data
    Datasets = {
        "qrecc": QReCCDataset,
        "cast19": CAsTDataset,
        "cast20": CAsTDataset,
        "topiocqa": TopiOCQADataset
    }
    train_dataset = Datasets[args.dataset](args, query_tokenizer, doc_tokenizer, args.train_file_path, need_doc_info=args.need_doc_info)
    
    # 在分布式训练环境中设置适当的批处理大小和采样器，
    # 以确保每个 GPU 都有足够的数据进行训练，并且在采样时能够避免数据的重复。
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.n_gpu > 1:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, 
                                batch_size=args.per_gpu_train_batch_size, 
                                sampler=sampler, 
                                collate_fn=train_dataset.get_collate_fn(args))
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    # 创建了一个优化器对象 optimizer。
    # get_optimizer 函数根据一些参数（可能包括学习率、权重衰减等）以及给定的模型（在这里是 query_encoder）返回一个优化器对象。
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)

    # 这一行创建了一个学习率调度器（scheduler）。get_linear_schedule_with_warmup 是 PyTorch 的学习率调度器函数之一。
    # 它使用线性学习率调度，包括一个预热阶段（warm-up phase）。
    # 在预热阶段，学习率逐渐增加，然后保持不变，这有助于训练的稳定性。
    # num_warmup_steps 参数指定了预热阶段的步数，num_training_steps 参数指定了总的训练步数。
    # 这两个参数通常与总的 epoch 数和批次大小一起使用，以便在整个训练过程中平滑调整学习率。
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_training_steps)


    # begin to train
    logger.info("Start training...")
    logger.info("Total training samples = {}".format(len(train_dataset)))
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    # 根据训练数据的大小和批次大小，将用户设置的与时间相关的步数（比如每多少步保存一次模型或打印一次日志）转换为与 epoch 相关的步数。
    # 这样可以确保在不同规模的数据集和不同的训练批次下，这些步骤的频率能够适应不同的情况。
    args.model_save_steps = max(1, int(args.model_save_steps * num_steps_per_epoch))
    args.log_print_steps = max(1, int(args.log_print_steps * num_steps_per_epoch))

    cur_step = 0

    # trange 更适用于整数范围的迭代，而 tqdm 更灵活，可以用于任何可迭代对象。
    epoch_iterator = trange(args.num_train_epochs, desc="Epoch")
    for epoch in epoch_iterator:
        query_encoder.train()

        # 在分布式训练环境中，通过设置每个 epoch 的采样器，确保每个 GPU 在每个 epoch 中都有不同的样本进行训练。
        # 这对于提高训练的多 GPU 效率和模型性能是有帮助的。
        if args.n_gpu > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader,  desc="Step"):
            query_encoder.zero_grad()


            if args.train_input_type == "flat_concat":
                input_ids = batch["bt_concat"].to(args.device)
                input_masks = batch["bt_concat_mask"].to(args.device)
            elif args.train_input_type == "concat_for_inversion":
                input_ids = batch["bt_concat_inversion"].to(args.device)
                input_masks = batch["bt_concat_mask_inversion"].to(args.device)
            else:
                raise ValueError("train input type:{}, has not been implemented.".format(args.train_input_type))
            
           
            concat_embs = query_encoder(input_ids, input_masks)  # B * dim
            
            if args.loss_type == "kd":
                oracle_query_encoder.eval()
                bt_oracle_utt = batch["bt_oracle_utt"].to(args.device)
                bt_oracle_utt_mask = batch["bt_oracle_utt_mask"].to(args.device)
                with torch.no_grad():
                # freeze oracle query encoder's parameters
                    oracle_utt_embs = oracle_query_encoder(bt_oracle_utt, bt_oracle_utt_mask).detach()  # B * dim
                loss = cal_kd_loss(concat_embs, oracle_utt_embs)
            elif args.loss_type == "ranking":
                bt_pos_docs = batch['bt_pos_docs'].to(args.device)
                bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
                bt_neg_docs = batch['bt_neg_docs'].to(args.device)
                bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
                with torch.no_grad():
                # doc encoder's parameters are frozen
                    pos_doc_embs = doc_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                    if len(batch['bt_neg_docs']) == 0:  # only_in_batch negative
                            neg_doc_embs = None
                    else:
                        batch_size, neg_ratio, seq_len = bt_neg_docs.shape   
                        # 将batch_size与neg_ratio合并为一个维度  
                        bt_neg_docs = bt_neg_docs.view(batch_size * neg_ratio, seq_len)        
                        bt_neg_docs_mask = bt_neg_docs_mask.view(batch_size * neg_ratio, seq_len)             
                        neg_doc_embs = doc_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # (B * neg_ratio) * dim,      
                loss = cal_ranking_loss(concat_embs, pos_doc_embs, neg_doc_embs)

            loss.backward()
            # 这一行代码用于梯度裁剪（gradient clipping）
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            # 用优化器（通常是梯度下降的一种变体）来更新模型参数。
            # 优化器通过利用损失函数的梯度信息，沿着梯度的负方向更新模型参数，以减小损失函数的值。
            optimizer.step()
            # 调整学习率。学习率调度器是一种动态地调整学习率的机制，可以根据训练的进展来自动调整学习率。
            # scheduler.step() 在每个训练步骤之后更新学习率。
            scheduler.step()
            
            # print info
            if dist.get_rank() == 0 and cur_step % args.log_print_steps == 0:
                logger.info("Epoch = {}, Current Step = {}, Total Step = {}, Loss = {}".format(
                                epoch,
                                cur_step,
                                total_training_steps,
                                round(loss.item(), 7))
                            )
            if dist.get_rank() == 0:
                log_writer.add_scalar("train_{}_loss".format(args.loss_type), loss, cur_step)
            cur_step += 1    # avoid saving the model of the first step.
            dist.barrier()
            # Save model
            if dist.get_rank() == 0 and args.need_output and cur_step % args.model_save_steps == 0:
                save_model(args.model_output_path, query_encoder, query_tokenizer, epoch)


    logger.info("Training finish!")          
    if dist.get_rank() == 0:   
        log_writer.close()
       

def get_args():
    parser = argparse.ArgumentParser()

    # 在分布式训练中，--local_rank 通常用于指定当前进程的本地排名。在 PyTorch 中，local_rank 是一个重要的参数，它告诉当前进程在分布式环境中的位置。
    # 每个进程都有一个唯一的本地排名，它表示该进程在整个分布式系统中的相对位置。
    # 在 PyTorch 的分布式训练中，你可以使用 torch.distributed.launch 或其他方法来启动多个进程，并通过 --local_rank 参数为每个进程分配本地排名。
    # 这对于确保每个进程知道自己在整个训练过程中的角色和位置非常重要，以便进行适当的通信和同步。
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser.add_argument('--n_gpu', type=int, default=1, help='The number of used GPU.')
    
    parser.add_argument("--dataset", type=str, required=True, choices=["cast19", "cast20", "qrecc", "topiocqa"])
    parser.add_argument("--model_type", type=str, required=True, choices=["ance", "dpr-nq", "tctcolbert","gtr-base","bge"])
    parser.add_argument("--pretrained_query_encoder_path", type=str, required=True, help="Path of the pretrained query encoder.")
    parser.add_argument("--pretrained_doc_encoder_path", type=str, required=True, help="Path of the pretrained doc encoder.")
    parser.add_argument("--train_file_path", type=str, required=True, help="Path of the training dialog file.")
    parser.add_argument("--log_path", type=str, required=True, help="Path of output tensorboard log.")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path of saved models.")
    parser.add_argument("--output_dir_path", type=str, required=True, help="Dir path of the output info.")
    parser.add_argument("--need_output", action="store_true", help="Whether need to output logs and models (creating the dirs)")
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")

    parser.add_argument("--log_print_steps", type=float, default=0.01, help="Percent of steps per epoch to print once.")
    parser.add_argument("--model_save_steps", type=float, required=True, help="Percent of steps to save the model once")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Training epochs")
    parser.add_argument("--per_gpu_train_batch_size", type=int, required=True, help="Per gpu batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Warm up steps.")

    parser.add_argument("--max_query_length", type=int, default=48, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, required=True, help="Max response length, 100 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, required=True, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--enable_last_response", action="store_true", help="True for CAsT-20")

    parser.add_argument("--loss_type", type=str, required=True, choices=["kd", "ranking"])
    parser.add_argument("--collate_fn_type", type=str, required=True, choices=["flat_concat_for_train", "flat_concat_for_test"], help="To control how to organize the batch data.")
    parser.add_argument("--train_input_type", type=str, required=True, choices=["flat_concat", "concat_for_inversion"], help="To choose which method to encode.")
   
    parser.add_argument("--negative_type", type=str, required=True, choices=["random_neg", "bm25_hard_neg", "prepos_hard_neg", "in_batch_neg"])
    parser.add_argument("--neg_ratio", type=int, help="negative ratio")
    parser.add_argument("--need_doc_info", action="store_true", help="Whether need doc info or not.")


    args = parser.parse_args()
    local_rank = args.local_rank
    args.local_rank = local_rank
    # pytorch parallel gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
    args.device = device
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    if dist.get_rank() == 0 and args.need_output:
        check_dir_exist_or_build([args.output_dir_path], force_emptying=args.force_emptying_dir)
        json_dumps_arguments(oj(args.output_dir_path, "parameters.txt"), args)
        

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    train(args)
