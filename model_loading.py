
import pickle
import torch
import transformers
import gradio as gr


# XLM model functions 
import transformers


# Our model definition 

class MultilingualClipEdited(torch.nn.Module):
    def __init__(self, model_name, tokenizer_name, head_name, weights_dir='head_weights/', cache_dir=None,in_features=None,out_features=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.head_path = weights_dir + head_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.transformer = transformers.AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.clip_head = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self._load_head()

    def forward(self, txt):
        txt_tok = self.tokenizer(txt, padding=True, return_tensors='pt')
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.clip_head(embs)

    def _load_head(self):
        with open(self.head_path, 'rb') as f:
            lin_weights = pickle.loads(f.read())
        self.clip_head.weight = torch.nn.Parameter(torch.tensor(lin_weights[0]).float().t())
        self.clip_head.bias = torch.nn.Parameter(torch.tensor(lin_weights[1]).float())

AVAILABLE_MODELS = {
    'bert-base-arabertv2-ViT-B-16-SigLIP-512-epoch-155-trained-2M':{
    'model_name': 'Arabic-Clip/bert-base-arabertv2-ViT-B-16-SigLIP-512-epoch-155-trained-2M',
    'tokenizer_name': 'Arabic-Clip/bert-base-arabertv2-ViT-B-16-SigLIP-512-epoch-155-trained-2M',
    'head_name': 'arabertv2-vit-B-16-siglibheads_of_the_model_arabertv2-ViT-B-16-SigLIP-512-155_.pickle'
    },

}


def load_model(name, cache_dir=None,in_features=None,out_features=None):
    config = AVAILABLE_MODELS[name]
    return MultilingualClipEdited(**config, cache_dir=cache_dir, in_features= in_features, out_features=out_features)
