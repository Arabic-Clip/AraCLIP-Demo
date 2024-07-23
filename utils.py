
import os 
import numpy as np
import pickle
import torch
import transformers
from PIL import Image
from open_clip import create_model_from_pretrained, create_model_and_transforms
import json 

# XLM model functions 
from multilingual_clip import pt_multilingual_clip

from model_loading import load_model



class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, compose, image_name_list):
        self.main_dir = main_dir
        self.transform = compose
        self.total_imgs = image_name_list

    def __len__(self):
        return len(self.total_imgs)

    def get_image_name(self, idx):

        return self.total_imgs[idx]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc)

        return self.transform(image)


def features_pickle(file_path=None):

    with open(file_path, 'rb') as handle:
        features_pickle = pickle.load(handle)

    return features_pickle


def dataset_loading(file_name):

    with open(file_name) as filino:

        data = [json.loads(file_i) for file_i in filino]

    sorted_data = sorted(data, key=lambda x: x['id'])

    image_name_list = [lin["image_name"] for lin in sorted_data]


    return sorted_data, image_name_list 


def text_encoder(language_model, text):
    """Normalize the text embeddings"""
    embedding = language_model(text)
    norm_embedding = embedding / np.linalg.norm(embedding)

    return embedding, norm_embedding


def compare_embeddings(logit_scale, img_embs, txt_embs):
  
  image_features = img_embs / img_embs.norm(dim=-1, keepdim=True)

  text_features = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

  logits_per_text = logit_scale * text_features @ image_features.t()

  return logits_per_text

# Done 
def compare_embeddings_text(full_text_embds, txt_embs):
  
  full_text_embds_features = full_text_embds / full_text_embds.norm(dim=-1, keepdim=True)

  text_features = txt_embs / txt_embs.norm(dim=-1, keepdim=True)

  logits_per_text_full = text_features @ full_text_embds_features.t()

  return logits_per_text_full



def find_image(language_model,clip_model, text_query, dataset, image_features, text_features_new,sorted_data, images_path,num=1):

    embedding, _  = text_encoder(language_model, text_query)

    logit_scale = clip_model.logit_scale.exp().float().to('cpu')

    language_logits, text_logits = {}, {}

    language_logits["Arabic"] = compare_embeddings(logit_scale, torch.from_numpy(image_features), torch.from_numpy(embedding))

    text_logits["Arabic_text"] = compare_embeddings_text(torch.from_numpy(text_features_new), torch.from_numpy(embedding))

    
    for _, txt_logits in language_logits.items():

        probs = txt_logits.softmax(dim=-1).cpu().detach().numpy().T

        file_paths = []
        labels, json_data = {}, {}

        for i in range(1, num+1):
            idx = np.argsort(probs, axis=0)[-i, 0]
            path = images_path + dataset.get_image_name(idx)
                    
            path_l = (path,f"{sorted_data[idx]['caption_ar']}")

            labels[f" Image # {i}"] = probs[idx]
            json_data[f" Image # {i}"] = sorted_data[idx]

            file_paths.append(path_l)


    json_text = {} 

    for _, txt_logits_full in text_logits.items():

        probs_text = txt_logits_full.softmax(dim=-1).cpu().detach().numpy().T

        for j in range(1, num+1):

            idx = np.argsort(probs_text, axis=0)[-j, 0]
            json_text[f" Text # {j}"] = sorted_data[idx]

    return file_paths, labels, json_data, json_text



class AraClip():
    def __init__(self):

        self.text_model = load_model('bert-base-arabertv2-ViT-B-16-SigLIP-512-epoch-155-trained-2M', in_features= 768, out_features=768)
        self.language_model = lambda queries: np.asarray(self.text_model(queries).detach().to('cpu')) 
        self.clip_model, self.compose = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')

        self.sorted_data_xtd, self.image_name_list_xtd = dataset_loading("photos/en_ar_XTD10_edited_v2.jsonl")
        self.sorted_data_flicker8k, self.image_name_list_flicker8k = dataset_loading("photos/flicker_8k.jsonl")

    def load_pickle_file(self, file_name):

        return features_pickle(file_name) 

    def load_xtd_dataset(self):
        dataset = CustomDataSet("photos/XTD10_dataset", self.compose, self.image_name_list_xtd)
        
        return dataset

    def load_flicker8k_dataset(self):
        dataset = CustomDataSet("photos/Flicker8k_Dataset", self.compose, self.image_name_list_flicker8k)
        return dataset

araclip = AraClip()

def predict(text, num, dadtaset_select):

    if dadtaset_select == "XTD dataset":
        image_paths, labels, json_data, json_text = find_image(araclip.language_model,araclip.clip_model, text, araclip.load_xtd_dataset(), araclip.load_pickle_file("cashed_pickles/XTD_pickles/image_features_XTD_1000_images_arabert_siglib_best_model.pickle") , araclip.load_pickle_file("cashed_pickles/XTD_pickles/image_features_XTD_1000_images_arabert_siglib_best_model.pickle"), araclip.sorted_data_xtd, 'photos/XTD10_dataset/', num=int(num))

    else:
        image_paths, labels, json_data, json_text = find_image(araclip.language_model,araclip.clip_model, text, araclip.load_flicker8k_dataset(), araclip.load_pickle_file("cashed_pickles/flicker_8k/image_features_flicker_8k_images_arabert_siglib_best_model.pickle") , araclip.load_pickle_file("cashed_pickles/flicker_8k/text_features_flicker_8k_images_arabert_siglib_best_model.pickle"), araclip.sorted_data_flicker8k, "photos/Flicker8k_Dataset/", num=int(num))

    return image_paths, labels, json_data, json_text


class Mclip():
    def __init__(self) -> None:

    
        self.tokenizer_mclip = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
        self.text_model_mclip = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
        self.language_model_mclip = lambda queries:  np.asarray(self.text_model_mclip.forward(queries, self.tokenizer_mclip).detach().to('cpu'))  
        self.clip_model_mclip, _, self.compose_mclip = create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
        self.sorted_data_xtd, self.image_name_list_xtd = dataset_loading("photos/en_ar_XTD10_edited_v2.jsonl")
        self.sorted_data_flicker8k, self.image_name_list_flicker8k = dataset_loading("photos/flicker_8k.jsonl")

    def load_pickle_file(self, file_name):

        return features_pickle(file_name) 


    def load_xtd_dataset(self):
        dataset = CustomDataSet("photos/XTD10_dataset", self.compose_mclip, self.image_name_list_xtd)
        
        return dataset

    def load_flicker8k_dataset(self):
        dataset = CustomDataSet("photos/Flicker8k_Dataset", self.compose_mclip, self.image_name_list_flicker8k)
        return dataset
    

mclip = Mclip()

def predict_mclip(text, num, dadtaset_select):


    if dadtaset_select == "XTD dataset":
        image_paths, labels, json_data, json_text = find_image(mclip.language_model_mclip,mclip.clip_model_mclip, text, mclip.load_xtd_dataset() , mclip.load_pickle_file("cashed_pickles/XTD_pickles/image_features_XTD_1000_images_XLM_Roberta_Large_Vit_B_16Plus_ar.pickle") , mclip.load_pickle_file("cashed_pickles/XTD_pickles/text_features_XTD_1000_images_XLM_Roberta_Large_Vit_B_16Plus_ar.pickle") , mclip.sorted_data_xtd , 'photos/XTD10_dataset/', num=int(num))

    else:
        image_paths, labels, json_data, json_text = find_image(mclip.language_model_mclip,mclip.clip_model_mclip, text, mclip.load_flicker8k_dataset() , mclip.load_pickle_file("cashed_pickles/flicker_8k/image_features_flicker_8k_images_XLM_Roberta_Large_Vit_B_16Plus_ar.pickle") , mclip.load_pickle_file("cashed_pickles/flicker_8k/text_features_flicker_8k_images_XLM_Roberta_Large_Vit_B_16Plus_ar.pickle") , mclip.sorted_data_flicker8k , 'photos/Flicker8k_Dataset/', num=int(num))

    
    return image_paths, labels, json_data, json_text
