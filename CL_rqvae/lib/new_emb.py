from re import X
import numpy as np
import pandas as pd
import torch



def take_feat(nlp_model_name, data_set_name, device, item_id_dict):
    device = torch.device("cuda:2")
    if nlp_model_name == 't5':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('pretrain/sentence-t5-base', )
        model = model.to(device)

    if data_set_name == "MIND":
        fileName = 'data/{}/news.tsv'.format(data_set_name)
    else: # Amazon
        fileName = 'data/{}/{}.item'.format(data_set_name, data_set_name)

    tsv_file = pd.read_csv(
        fileName,
        # converters={'strings': str},
        sep='\t',
        header=None,
        keep_default_na=False
        # dtype=str
    )
    tsv_file = tsv_file.fillna('unknown')
    if data_set_name == "MIND":
        item_num = len(tsv_file.values[:,0])#32489
    else:
        item_num = len(tsv_file.values[1:,0])# Amazon has title

    print(".item file length {}".format(item_num))

    if data_set_name == "Beauty":
        id = tsv_file.values[1:, 0]
        brand = tsv_file.values[1:, 2]
        sales_type = tsv_file.values[1:, 3]
        sales_rank = tsv_file.values[1: 4]
        title = tsv_file.values[1:, 1]
    elif data_set_name == "Health":
        id = tsv_file.values[1:, 0]
        title = tsv_file.values[1:, 1]
        sales_type = tsv_file.values[1:, 2]
    elif data_set_name == "Toys":
        id = tsv_file.values[1:, 0]
        title = tsv_file.values[1:, 1]
        sales_type = tsv_file.values[1:, 3]
        brand = tsv_file.values[1:, 5]
        cate = tsv_file.values[1:, 6]
    elif data_set_name == "Sports":
        id = tsv_file.values[1:, 0]
        title = tsv_file.values[1:, 1]
        brand = tsv_file.values[1:, 3]
        cate = tsv_file.values[1:, 4]
        sales_type = tsv_file.values[1:, 5]
    elif data_set_name == 'Sports_and_Outdoors' or data_set_name == 'Toys_and_Games':
        id = tsv_file.values[1:, 0]
        title = tsv_file.values[1:, 1]
        cate = tsv_file.values[1:, 2]
        brand = tsv_file.values[1:, 3]
        sales_type = tsv_file.values[1:, 4]
        sales_rank = tsv_file.values[1: 5]
    elif data_set_name == 'MIND':
        id = tsv_file.values[:, 0]
        type = tsv_file.values[:, 1]
        subtype = tsv_file.values[:, 2]
        title = tsv_file.values[:, 3]
        info = tsv_file.values[:, 4]
    elif data_set_name == 'Amazon_Office_Products':
        id = tsv_file.values[1:, 0]
        sales_type = tsv_file.values[1:, 2]
        brand = tsv_file.values[1:, 6]
        cate = tsv_file.values[1:, 4] 
        title = tsv_file.values[1:, 5]

    save_len = len(item_id_dict.keys())
    final_feat = np.zeros([save_len, 768])
    print("save_len:",save_len)

    sent_feat = np.zeros([1000000, 768])

    if data_set_name == 'MIND':
        sent_list = []
        sent_klist = []
        for i, id_ in enumerate(id):
            k = int(id_[1:])
            if k in item_id_dict.keys():
                v = item_id_dict[k] # k, v
                type_sent = "The type is " + type[i]
                sub_type_sent = "the subtype is " + subtype[i]
                title_sent = "the title is " + title[i]
                info_sent = 'other information is ' + str(info[i])
                sentences = type_sent + ', and ' + sub_type_sent + ', and ' + title_sent + ', and ' + info_sent + '.'
                sent_list.append(sentences)
                sent_klist.append(v)
        embeddings = model.encode(sent_list)
        print(embeddings.shape)
        for i in range(save_len):
            k = sent_klist[i]
            v = embeddings[i]
            final_feat[k] = v

        feat_path = 'data/{}/{}_{}'.format(data_set_name, data_set_name, nlp_model_name)

        print("save feature path ", feat_path)
        np.save(feat_path, final_feat)

        result = []
        for idx, z in enumerate(final_feat):
            if z.sum() == 0:
                result.append(idx)

        print("embeddings=0", len(result))

        return final_feat

    elif data_set_name == 'Toys' or data_set_name == 'Sports' or  data_set_name == 'Amazon_Office_Products':
        sent_list = []
        sent_klist = []
        for i, id_ in enumerate(id):
            k = id_
            if k in item_id_dict.keys():
                v = item_id_dict[k] # k, v # sales_type + brand + cate + title
                sales_type_sent = "The type is " + sales_type[i]
                brand_sent = "the brand is " + brand[i]
                title_sent = "the title is " + title[i]
                cate_sent = 'the category is ' + str(cate[i])
                sentences = sales_type_sent + ', and ' + title_sent + ', and ' + brand_sent + '.'#+ ', and ' + cate_sent + '.'
                sent_list.append(sentences)
                sent_klist.append(v)
        embeddings = model.encode(sent_list)

        for i in range(save_len):
            k = sent_klist[i]
            v = embeddings[i]
            final_feat[k] = v
        feat_path = 'data/{}/{}_{}_{}'.format(data_set_name, data_set_name, nlp_model_name, embeddings.shape[1])

        print("save feature path ", feat_path)
        np.save(feat_path, final_feat)

        result = []
        for idx, z in enumerate(final_feat):
            if z.sum() == 0:
                result.append(idx)

        print("embeddings=0", len(result))

        return final_feat
        
    elif data_set_name == 'Health':
        sent_list = []
        sent_klist = []
        for i, id_ in enumerate(id):
            k = id_
            if k in item_id_dict.keys():
                v = item_id_dict[k] # k, v
                sales_type_sent = "The type is " + sales_type[i]
                title_sent = "the title is " + title[i]
                sentences = sales_type_sent + ', and ' + title_sent + '.' 
                #+ ', and ' + brand_sent + '.'#+ ', and ' + cate_sent + '.'
                sent_list.append(sentences)
                sent_klist.append(v)
        embeddings = model.encode(sent_list)

        for i in range(save_len):
            k = sent_klist[i]
            v = embeddings[i]
            final_feat[k] = v
        feat_path = 'data/{}/{}_{}_{}'.format(data_set_name, data_set_name, nlp_model_name, embeddings.shape[1])

        print("save feature path ", feat_path)
        np.save(feat_path, final_feat)

        result = []
        for idx, z in enumerate(final_feat):
            if z.sum() == 0:
                result.append(idx)

        print("embeddings=0", len(result))

        return final_feat



device = 'cuda:3'
if device!='cpu':
    device = torch.device(device)
data_set_name = 'MIND'
item_dict_path = 'data/{}/item_dict'.format(data_set_name)
item_id_dict = np.load(item_dict_path+'.npy', allow_pickle=True).item()
take_feat('t5', data_set_name, device, item_id_dict)
