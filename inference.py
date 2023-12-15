from data_load import HParams
from model import Net
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert import BertModel
import parameters
import numpy as np 
import torch
import sys
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree

config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
def build_model(config, state_dict, hp):
    model = Net(config, vocab_len = len(hp.VOCAB), bert_state_dict=None)
    _ = model.load_state_dict(torch.load(state_dict, map_location='cpu'))
    _ = model.to('cpu')  # inference 
    return model 


# Model loaded 
bc5_model = build_model(config, parameters.BC5CDR_WEIGHT, HParams('bc5cdr'))


# Process Query 
def process_query(query, hp, model):
    s = query
    split_s = ["[CLS]"] + s.split()+["[SEP]"]
    x = [] # list of ids
    is_heads = [] # list. 1: the token is the first piece of a word

    for w in split_s:
        tokens = hp.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        xx = hp.tokenizer.convert_tokens_to_ids(tokens)
        is_head = [1] + [0]*(len(tokens) - 1)
        x.extend(xx)
        is_heads.extend(is_head)

    x = torch.LongTensor(x).unsqueeze(dim=0)

    # Process query 
    model.eval()
    _, _, y_pred,embedding = model(x, torch.Tensor([1, 2, 3]))  # just a dummy y value
    preds = y_pred[0].cpu().numpy()[np.array(is_heads) == 1]  # Get prediction where head is 1 

    # convert to real tags and remove <SEP> and <CLS>  tokens labels 
    preds = [hp.idx2tag[i] for i in preds][1:-1]
    final_output = []
    print(embedding.shape)
    for word, label in zip(s.split(), preds):
        final_output.append([word, label])
    return final_output

def merge_BIO_tags_to_entities(output):
    tokens=[]
    tags=[]
    #store the sentence and label.
    for i,j in output:
        tokens.append(i)
        tags.append(j)
    # tag each token with its part of speech.
    pos_tags = [pos for token, pos in pos_tag(tokens)]
    # We convert the BIO / IOB tags to tree
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    # Parse the tree to get our original text
    original_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])
            original_text.append((original_string, original_label))
    return original_text
#     print(original_text[0])
#     print(type(original_text[0]))
# print(output)
# merge_BIO_tags_to_entities(output)

def get_ner(query):
    hp = HParams('bc5cdr')
    # print(hp.VOCAB)
    # print(hp.tag2idx) 
    # print(hp.idx2tag)
    print("bc5cdr -> ", query)
    output = process_query(query=query, hp=hp, model=bc5_model)
    # return JSONResponse({'tagging': out})
    ners = merge_BIO_tags_to_entities(output)
    return ners

if __name__=="__main__":
    query = input("Enter the clinical notes: ")
    output = get_ner(query)
