from IPython import embed

import os
import json
import shutil
import pickle
import random
import numpy as np
from os.path import join as oj
import pytrec_eval
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from pprint import pprint
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AdamW


# from multiprocessing import Process


def check_dir_exist_or_build(dir_list, force_emptying: bool = False):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
        elif len(os.listdir(x)) > 0:  # not empty
            if force_emptying:
                print("Forcing to erase all contens of {}".format(x))
                shutil.rmtree(x)
                os.makedirs(x)
            else:
                raise FileExistsError
        else:
            continue


def json_dumps_arguments(output_path, args):
    with open(output_path, "w") as f:
        params = vars(args)
        if "device" in params:
            params["device"] = str(params["device"])
        f.write(json.dumps(params, indent=4))


def split_and_padding_neighbor(batch_tensor, batch_len):
    batch_len = batch_len.tolist()
    pad_len = max(batch_len)
    device = batch_tensor.device
    tensor_dim = batch_tensor.size(1)

    batch_tensor = torch.split(batch_tensor, batch_len, dim=0)

    padded_res = []
    for i in range(len(batch_tensor)):
        cur_len = batch_tensor[i].size(0)
        if cur_len < pad_len:
            padded_res.append(torch.cat([batch_tensor[i],
                                         torch.zeros((pad_len - cur_len, tensor_dim)).to(device)], dim=0))
        else:
            padded_res.append(batch_tensor[i])

    padded_res = torch.cat(padded_res, dim=0).view(len(batch_tensor), pad_len, tensor_dim)

    return padded_res


def get_has_gold_label_test_qid_set(qrel_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    qids = set()
    for line in qrel_data:
        record = line.strip().split("\t")
        if len(record) != 4:
            record = line.strip().split(" ")
        query = record[0]
        qids.add(query)
    return qids


def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res


def pstore(x, path, high_protocol=False):
    with open(path, 'wb') as f:
        if high_protocol:
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


def load_collection(collection_file):
    all_docs = {}
    with open(collection_file, "r") as f:
        for line in f:
            line = line.strip()
            try:
                line_arr = line.split("\t")
                pid = int(line_arr[0])
                # 去除末尾的空白字符
                passage = line_arr[1].rstrip()
                all_docs[pid] = passage
            except IndexError:
                print("bad passage")
            except ValueError:
                print("bad pid")
    return all_docs


def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)


def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec


def barrier_array_merge(args,
                        data_array,
                        merge_axis=0,
                        prefix="",
                        load_cache=False,
                        only_load_in_master=False,
                        merge=True):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    if not merge:
        return None

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(args.world_size
                   ):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


# for gpt/t5 inference
def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "then", "here", "there", "when", "where", "why", "how", "other", "some", "such", "nor", "not", "only",
              "own", "same", "now"]
BIG_STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                  "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                  "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                  "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                  "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                  "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                  "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
                  "now"]
BIGBIG_STOP_WORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about",
                     "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad",
                     "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag",
                     "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone",
                     "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount",
                     "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything",
                     "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate",
                     "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as",
                     "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away",
                     "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became",
                     "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning",
                     "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best",
                     "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom",
                     "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca",
                     "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain",
                     "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn",
                     "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering",
                     "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't",
                     "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz",
                     "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite",
                     "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does",
                     "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt",
                     "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef",
                     "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em",
                     "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially",
                     "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone",
                     "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far",
                     "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five",
                     "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly",
                     "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further",
                     "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given",
                     "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings",
                     "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt",
                     "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help",
                     "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers",
                     "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home",
                     "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i",
                     "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig",
                     "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance",
                     "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates",
                     "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io",
                     "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself",
                     "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep",
                     "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely",
                     "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest",
                     "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln",
                     "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made",
                     "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile",
                     "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo",
                     "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must",
                     "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne",
                     "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither",
                     "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no",
                     "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted",
                     "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain",
                     "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay",
                     "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or",
                     "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves",
                     "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3",
                     "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc",
                     "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm",
                     "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly",
                     "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides",
                     "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra",
                     "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref",
                     "refs", "regarding", "regardless", "regards", "related", "relatively", "research",
                     "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right",
                     "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa",
                     "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly",
                     "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
                     "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't",
                     "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've",
                     "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly",
                     "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn",
                     "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes",
                     "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify",
                     "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially",
                     "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1",
                     "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th",
                     "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the",
                     "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
                     "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto",
                     "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're",
                     "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou",
                     "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti",
                     "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards",
                     "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve",
                     "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under",
                     "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur",
                     "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v",
                     "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols",
                     "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't",
                     "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren",
                     "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence",
                     "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's",
                     "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever",
                     "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely",
                     "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't",
                     "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf",
                     "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl",
                     "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves",
                     "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
QUESTION_WORD_LIST = ["what", "when", "why", "who", "how", "where", "whose", "which", "is", "are", "were", "was", "do",
                      "does", "did", "can"]
OTHER_WORD_LIST = ["tell"]


def is_nl_query(query):
    if any([query.lower().startswith(word) for word in QUESTION_WORD_LIST]):
        return True
    return False


def format_nl_query(query):
    query = query.replace("?", "")
    query = query.replace("\\", "")
    query = query.replace("\"", "")
    if is_nl_query(query):
        query = query[0].upper() + query[1:] + "?"
    else:
        query = query[0].upper() + query[1:] + "."
    return query

def get_has_qrel_label_sample_ids(qrel_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    qids = set()
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        qids.add(query)
    return qids


def eval_run_with_qrel(**eval_kwargs):
    rel_threshold = eval_kwargs["rel_threshold"] if "rel_threshold" in eval_kwargs else 1
    retrieval_output_path = eval_kwargs["retrieval_output_path"] if "retrieval_output_path" in eval_kwargs else "./"

    if "run" in eval_kwargs:
        runs = eval_kwargs["run"]
    else:
        assert "run_file" in eval_kwargs
        with open(eval_kwargs["run_file"], 'r' )as f:
            run_data = f.readlines()
        runs = {}
        for line in run_data:
            line = line.split(" ")
            sample_id = line[0]
            pid = line[2]
            rel = float(line[4])
            if sample_id not in runs:
                runs[sample_id] = {}
            runs[sample_id][pid] = rel

    assert "qrel_file" in eval_kwargs
    with open(eval_kwargs["qrel_file"], 'r') as f:
        qrel_data = f.readlines()

    qrels = {}
    qrels_ndcg = {}
    for line in qrel_data:
        lines = line.strip().split("\t")
        if len(lines) == 1:
            lines = line.strip().split(" ")
        query = lines[0]
        passage = lines[2]
        rel = int(lines[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
 
    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    print(map_list)
    print(ndcg_3_list)
    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
            "NDCG@3": np.average(ndcg_3_list), 
        }

    
    print("---------------------Evaluation results:---------------------")    
    pprint(res)
    with open(os.path.join(retrieval_output_path, "res.txt"), "w") as f:
        f.write(json.dumps(res, indent=4))

    return res


def agg_res_with_maxp(run_trec_file):
    with open(run_trec_file, 'r' ) as f:
        run_data = f.readlines()
    
    agg_run = {}
    for line in run_data:
        line = line.strip().split(" ")
        if len(line) == 1:
            line = line.strip().split('\t')
        sample_id = line[0]
        if sample_id not in agg_run:
            agg_run[sample_id] = {}
        doc_id = "_".join(line[2].split('_')[:2])
        score = float(line[4])

        if doc_id not in agg_run[sample_id]:
            agg_run[sample_id][doc_id] = 0
        agg_run[sample_id][doc_id] = max(agg_run[sample_id][doc_id], score)
    
    agg_run = {k: sorted(v.items(), key=lambda item: item[1], reverse=True) for k, v in agg_run.items()}
    with open(os.path.join(run_trec_file + ".agg"), "w") as f:
        for sample_id in agg_run:
            doc_scores = agg_run[sample_id]
            rank = 1
            for doc_id, real_score in doc_scores:
                rank_score = 1000 - rank
                f.write("{} Q0 {} {} {} {}\n".format(sample_id, doc_id, rank, rank_score, real_score, "ance"))
                rank += 1   




