import torch
import sys
import math
from lib.Trm4Rec_trainer import Trm4Rec
from lib.preprocess import preprocess_data_mind, preprocess_data_amazon
import numpy as np
from lib.generate_training_batches import Train_instance
from pandas import DataFrame
import logging
from lib.new_emb import take_feat
from lib.train_rqvae import train_rqvae
from lib.matrix import presision, recall, f_measure, novelty, hit_ratio, NDCG, MAP, NDCG_comicrec, NDCG_recforest

# conda activate jmq
# cd RecForest
# python train_rec.py

# 创建一个 logger
logger = logging.getLogger('my_logger')

sys.path.append('../..')
#parametres
data_set_name = 'MIND'
# 'MIND' # 'Sports_and_Outdoors' 'Amazon_Office_Products'# 'Beauty'#'Health' 'gowalla' 'Toys_and_Games'
optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True)
have_processed_data = False
have_processed_code = False
have_processed_feat = False
tree_has_generated = False

init_way=  'embrqvae' #'random' 
# code的编码方式，'embrqvae'是tiger的编码；'embkm'是recforest的kmeans clustering方法；'random'是随机编码
#'embrqvae' # 'embkm' 'random' 
tree_num= 1
num_layers=1 #layer of transformer
k= 64 #16 #32 #36 #the branch of each tree
n_head=4
d_model=96
nlp_model_name = 't5'
rqvae_model_name = 'rq'
device='cuda:3'
rerank_topk= [5, 10, 20, 40]  # 想得到前topk个结果
topk= 40 # 大的那个 predict取决于这个
# -------------训练rqvae的参数---------------


parall=10
seq_len= 50 #21 # se_len-1 is the number of behaviours in all the windows  16 #
min_seq_len= 3 #15#3

if data_set_name == "MIND":
    test_user_num = 10000
else:
    test_user_num=0 # the number of user in test file
    ratingbar = 1.0 # amazon数据集有一个rating打分的bar

max_iters=100
feature_ratio=1.0
reranker="Trm"
total_batch_num=50000
test_batch_size=100

train_instances_file='data/{}/train_instances'.format(data_set_name)
test_instances_file='data/{}/test_instances'.format(data_set_name)
validation_instances_file='data/{}/val_instances'.format(data_set_name)
item_num_node_num_file='data/{}/item_node_num.txt'.format(data_set_name)
train_item_vec_file='data/{}/train_item_vec.npy'.format(data_set_name)
#item_to_code_file='../../data/{}/item_to_code.npy'.format(data_set_name)
#code_to_item_file='../../data/{}/code_to_item.npy'.format(data_set_name)
DIN_Model_path='data/{}/DIN_MODEL_60000.pt'.format(data_set_name)
raw_inter_file = 'data/{}/{}.inter'.format(data_set_name, data_set_name)
if data_set_name == "Sports_and_Outdoors" or data_set_name =='Beauty'or data_set_name =='Toys_and_Games':
    raw_item_file = 'data/{}/{}.item'.format(data_set_name, data_set_name)
elif data_set_name == 'MIND':
    raw_item_file = 'data/{}/news.tsv'.format(data_set_name)
item_dict_path = 'data/{}/item_dict'.format(data_set_name)

filename = 'log/{}_{}_{}'.format(data_set_name, nlp_model_name, init_way)
logging.basicConfig(level=logging.INFO, filename = filename,  filemode = 'a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info('have_processed_data{}, have_processed_feat{}, have_processed_code{}, tree_has_generated{}'.format(have_processed_data, have_processed_feat, have_processed_code,tree_has_generated))
logging.info('init_way{}, tree_num{}, num_layers{}, k{}, n_head{}, d_model{}'.format(init_way,tree_num,num_layers,k,n_head, d_model))
logging.info('nlp_model_name{}, rqvae_model_name{}, device{}, rerank_topk{}, topk{}, distill{}'.format(nlp_model_name, rqvae_model_name, device, rerank_topk, topk, distill))
logging.info('parall{}, seq_len{}, min_seq_len{}, test_user_num{}'.format(parall, seq_len, min_seq_len, test_user_num))
logging.info('max_iters{}, feature_ratio{}, reranker{}, total_batch_num{}, test_batch_size{}'.format(max_iters, feature_ratio, reranker, total_batch_num, test_batch_size))

item_to_code_file_list=[]
code_to_item_file_list=[]

for tree_id in range(tree_num):
    item_to_code_file='data/{}/tree/{}{}_t5_item_to_code_tree_id_{}_k{}.npy'.format(data_set_name,init_way,feature_ratio,tree_id,k)
    code_to_item_file='data/{}/tree/{}{}_t5_code_to_item_tree_id_{}_k{}.npy'.format(data_set_name,init_way,feature_ratio,tree_id,k)
    item_to_code_file_list.append(item_to_code_file)
    code_to_item_file_list.append(code_to_item_file)
#assert tree_num==1

eps=0.00001
if device!='cpu':
    device = torch.device(device)

if not have_processed_data:
    if data_set_name == 'MIND':
        preprocess_data_mind(data_set_name, seq_len, min_seq_len)
    else:
        preprocess_data_amazon(data_set_name, seq_len, min_seq_len, ratingbar)
# 已写入了train_instances, val_instances, test_instances, item_dict.npy文件，读出来
[user_num,item_num]=np.loadtxt(item_num_node_num_file,dtype=np.int32,delimiter=',')
item_id_dict = np.load(item_dict_path+'.npy', allow_pickle=True).item()
print('user num is {}, item is {}'.format(user_num,item_num))


train_instances=Train_instance(parall=parall)
#training_batch_generator=train_instances.training_batches(train_instances_file,train_sample_seg_cnt,item_num,batchsize=training_batch_size)
his_maxtix_path = 'data/{}/his_maxtix.pt'.format(data_set_name)
labels_path = 'data/{}/labels.pt'.format(data_set_name)
training_data,training_labels=train_instances.get_training_data(train_instances_file, item_num)#,his_maxtix_path, labels_path, his_maxtix=None, labels=None)
#test_batch_generator=train_instances.test_batches(test_instances_file,item_num,batchsize=test_batch_size)
print("number training data is {}, label len {}".format(training_data.shape, len(training_labels)))
validation_batch_generator=train_instances.validation_batches(validation_instances_file,item_num,batchsize=test_batch_size)
test_instances=train_instances.read_test_instances_file(test_instances_file,item_num)
# print("val num {}, test num {}".format(len(validation_batch_generator), len(test_instances)))
print("number test data is {}".format(test_instances.shape))

moving_average = lambda x, **kw: DataFrame({'x':np.asarray(x)}).x.ewm(**kw).mean().values
loss_history,dev_precision_history,dev_recall_history,dev_f_measure_history,dev_novelty_history,policy_acc,dev_hr_history,dev_ndcg_history,dev_map_history=[],[],[],[],[],[],[],[],[]
total_precision_history,total_recall_history,total_f_measure_history,total_novelty_history,total_hr_history,total_ndcg_history,total_map_history=[],[],[],[],[],[],[]

if not have_processed_feat:
    final_feat = take_feat(nlp_model_name, data_set_name, device, item_id_dict)
else:
    path1 = 'data/{}/{}_{}.npy'.format(data_set_name, data_set_name, nlp_model_name)
    final_feat = np.load(path1, allow_pickle=True)

# 如果已经处理好code了，载入code; 如果没处理好code，embrqvae方法去训练rqvae生成code，random方法没有codes
if have_processed_code:
    if init_way == 'embrqvae':
        path = 'data/{}/{}.pth'.format(data_set_name, rqvae_model_name)
        state_dict_codes = torch.load(path)
        codes = state_dict_codes['codes']
else:
    if init_way == 'embrqvae':
        rqvae_config = {'batch_size':256, 'num_quantizers':3, 'C':768, 'D':96, 'codebook_size': 64, 'cos_add':True, 'total_epoch':20, 'distill':True, 'mse': False}
        codes = train_rqvae(data_set_name, rqvae_model_name, final_feat, rqvae_config, device)
        print(codes)
    else:
        codes = None
data = final_feat


train_model_list = []
for i in range(tree_num):
    train_model = Trm4Rec(item_num=int(item_num),
                          user_seq_len=seq_len-1,
                          d_model=d_model,
                          nhead=n_head,
                          device=device,
                          optimizer=optimizer,
                          num_layers=num_layers,
                          k=k,
                          item_to_code_file=item_to_code_file_list[i],
                          code_to_item_file=code_to_item_file_list[i],
                          tree_has_generated=tree_has_generated,
                          init_way=init_way,
                          max_iters=max_iters,
                          feature_ratio=feature_ratio,
                          data=data,#used for kmeans tree
                          parall=parall,
                          codes = codes)
    if i > 0:
        train_model.trm_model.trm.encoder = train_model_list[0].trm_model.trm.encoder
    train_model_list.append(train_model)

model_parameters = list(train_model_list[0].trm_model.trm.encoder.parameters())
for i in range(tree_num):
    model_parameters += list(train_model_list[i].trm_model.trm.decoder.parameters())
model_optimizer = optimizer(model_parameters)


num_batch = math.ceil(test_instances.shape[0] / test_batch_size)

for i in range(tree_num):
    train_model_list[i].trm_model.train()

def rerank(batch_x, label, top_k=rerank_topk, tree_num=11):
    if tree_num == 1:
        return label
    scores = torch.full((len(batch_x), top_k * tree_num), -1e15, device=device)
    input_labels = torch.zeros((len(batch_x), top_k * tree_num), dtype=torch.int64, device=device)
    max_lenr = top_k
    for i, user, result in zip(range(len(batch_x)), batch_x, label):
        r = torch.LongTensor(list(set(result.tolist())))
        scores[i, 0:len(r)] = 0.0
        input_labels[i, 0:len(r)] = r
        max_lenr = max(max_lenr, len(r))
    scores = scores[:, 0:max_lenr]
    input_labels = input_labels[:, 0:max_lenr]
    input_user = batch_x.repeat_interleave(max_lenr, dim=0)
    input_item = input_labels.reshape(-1)
    with torch.no_grad():
        for j in range(tree_num):
            scores += train_model_list[j].compute_scores( \
                input_user, \
                input_item).sum(-1).view(batch_x.shape[0], -1)
        argindex = scores.argsort(-1, True)[:, :top_k]
        final_result = input_labels.gather(index=argindex, dim=-1)
    return final_result

for (batch_x, batch_y) in train_instances.generate_training_records(training_data, training_labels, batch_size=512):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    loss = 0
    for i in range(tree_num):
        loss += train_model_list[i].update_model(batch_x, batch_y)
    loss.backward()
    model_optimizer.step()
    model_optimizer.zero_grad()
    loss_history.append(loss.item())
    if train_model_list[0].batch_num % 1000 == 0:
        print("------------------------test------------------------")
        for i in range(tree_num):
            train_model_list[i].trm_model.eval()
        result_history = []
        
        for j in range(num_batch):
            batch_result_list = []
            batch_user = test_instances[j * test_batch_size:(j + 1) * test_batch_size].to(device)
            for i in range(tree_num):
                batch_result_one_tree = train_model_list[i].predict(batch_user, topk=topk)
                batch_result_list.append(batch_result_one_tree)
            batch_result = torch.cat(batch_result_list, dim=-1)
            batch_result = rerank(batch_user, batch_result, rerank_topk, tree_num)
            result_history.append(batch_result)
        result_history = torch.cat(result_history, dim=0).cpu().numpy()
        print(result_history.shape)
        total_precision_history.append(presision(result_history, train_instances.test_labels, rerank_topk))
        total_recall_history.append(recall(result_history, train_instances.test_labels, rerank_topk))
        # total_f_measure_history.append(f_measure(result_history, train_instances.test_labels, rerank_topk))
        # total_novelty_history.append(novelty(result_history, test_instances.tolist(), rerank_topk))
        total_hr_history.append(hit_ratio(result_history, train_instances.test_labels, rerank_topk))
        total_ndcg_history.append(NDCG(result_history, train_instances.test_labels, rerank_topk))
        # total_map_history.append(MAP(result_history, train_instances.test_labels, rerank_topk))
        
        print(total_ndcg_history[-1])
        print(total_recall_history[-1])
        logging.info(total_ndcg_history[-1])
        logging.info(total_recall_history[-1])

        for i in range(tree_num):
            train_model_list[i].trm_model.train()

    if train_model_list[0].batch_num % 1000 == 0:
        print("------------------------val------------------------")
        for i in range(tree_num):
            train_model_list[i].trm_model.eval()
        test_batch, test_index = validation_batch_generator.__next__()
        print(test_batch.shape, test_index.shape)
        test_batch = test_batch.to(device)
        gt_history = [train_instances.validation_labels[i.item()] for i in test_index]
        # result_history=train_model.predict(test_batch,topk=topk).numpy()
        result_history = []
        for i in range(tree_num):
            result_history.append(train_model_list[i].predict(test_batch, topk=topk))
        result_history = rerank(test_batch, torch.cat(result_history, dim=-1), tree_num=tree_num).cpu().numpy()
        dev_precision_history.append(presision(result_history, gt_history, rerank_topk))
        dev_recall_history.append(recall(result_history, gt_history, rerank_topk))
        # dev_f_measure_history.append(f_measure(result_history, gt_history, rerank_topk))
        # dev_novelty_history.append(novelty(result_history, test_batch.tolist(), rerank_topk))
        # dev_hr_history.append(hit_ratio(result_history, gt_history, rerank_topk))
        dev_ndcg_history.append(NDCG(result_history, gt_history, rerank_topk))
        # dev_map_history.append(MAP(result_history, gt_history, rerank_topk))
        print(dev_ndcg_history[-1])
        print(dev_recall_history[-1])
        logging.info(dev_ndcg_history[-1])
        logging.info(dev_recall_history[-1])

        for i in range(tree_num):
            train_model_list[i].trm_model.train()

        x = "step=%i, mean_loss=%.3f" %(len(loss_history), np.mean(loss_history[-100:]))
        logging.info(x)
        print(x)

    if train_model_list[0].batch_num > total_batch_num:
        break

