import torch
import math
import numpy as np
from lib.KmeansTree import ConstructKmeansTree
class Tree:
    def __init__(self,data=None,max_iters=100,feature_ratio=0.8,item_num=1000, k=2,init_way='random',construct=True,parall=4, codes = None, device = 'cpu'):
        self.item_num = item_num
        self.k = k
        self.codes =codes
        self.device = device
        if construct:
            self.init_tree(data=data,max_iters=max_iters,init_way=init_way,feature_ratio=feature_ratio,parall=parall)

    def init_tree(self, data=None,max_iters=100,init_way='random',feature_ratio=0.8,parall=4):
        #data is item embeding or maxtirx used for kmeans cluster
        self.tree_height = math.ceil(math.log(self.item_num,self.k))#root node is rooted on layer zero, the leaf lyaer id
        num_all_leaf_code = self.tree_height * self.k ** self.tree_height#the number of leaf node
        print(self.k, self.tree_height, num_all_leaf_code)
        self.code_to_item = torch.zeros((num_all_leaf_code,),dtype=torch.int64)#node code to node
        self.item_to_code = {item_id:[] for item_id in range(self.item_num)}#record the path of item, one item can have multiple paths
        if init_way.lower()=='random':
            num_k_code_item = (num_all_leaf_code - self.item_num) // (self.k-1)
            item_seq=np.arange(self.item_num)
            np.random.shuffle(item_seq)
            start_code = 0
            for i,item_id in enumerate(item_seq):
                path = [start_code % (self.k ** (j+1)) // (self.k ** j) for j in range(self.tree_height-1,-1,-1)]
                if i < num_k_code_item:
                    for j in range(self.k):
                        path[-1] = j
                        self.item_to_code[item_id].append(torch.LongTensor(path))
                    self.code_to_item[start_code:start_code+self.k] = item_id
                    start_code = start_code + self.k
                else:
                    self.item_to_code[item_id].append(torch.LongTensor(path))
                    self.code_to_item[start_code:start_code+1] = item_id
                    start_code = start_code + 1
            item_id = item_seq[-1]
            while start_code < num_all_leaf_code:
                path = [start_code % (self.k ** (j+1)) // (self.k ** j) for j in range(self.tree_height-1,-1,-1)]
                self.item_to_code[item_id].append(torch.LongTensor(path))
                self.code_to_item[start_code]=item_id
                start_code = start_code + 1

            self.card = torch.zeros(self.tree_height)
            for i in range(self.tree_height):
                self.card[i] = self.k ** (self.tree_height - i - 1)# recover to the code
            self.card = self.card.to('cuda')
            self.code_to_item = self.code_to_item.to('cuda')

        elif init_way.lower() == 'embrqvae':
            self.tree_height = self.codes.shape[1] # rqvqe layer
            num_all_leaf_code =  self.tree_height * self.k ** (self.tree_height) #3 * 256 * 256 * 256  #self.k ** (self.tree_height - 1) +  # the number of leaf node
            print("num_all_leaf_code:",num_all_leaf_code)
            self.code_to_item = torch.zeros((num_all_leaf_code,), dtype=torch.int64)
            # self.code_to_item = {} 
            self.item_to_code = {item_id: [] for item_id in range(self.item_num)}  # record the path
            print('start to construct for rqvae')

            index = torch.arange(self.item_num)
            leaf_node_codes = self.codes
            for i, code in zip(range(len(index)), leaf_node_codes):
                real_item_id = index[i].item()
                reverse_path = []
                for j in range(self.tree_height):
                    reverse_path.append(((code[j]) % self.k).item())
                self.item_to_code[real_item_id].append(torch.LongTensor(reverse_path))
                unique_code = 0
                for j in range(self.tree_height):
                    unique_code += (self.k ** j * code[self.tree_height-1-j].item())

                self.code_to_item[int(unique_code)] = real_item_id
                # self.code_to_item.update({int(unique_code):real_item_id})
            
            self.card = torch.zeros(self.tree_height)
            # for i in range(self.tree_height):
            #     self.card[i] = self.k ** (self.tree_height - i - 1)  # recover to the code
            for i in range(0, self.tree_height):
                self.card[i] = self.k ** (self.tree_height - (i + 1))

            # for i in self.item_to_code:

            # self.card[-1] = self.k ** (self.tree_height - 1)
            # self.card = self.card.to('cuda')
            # self.code_to_item = self.code_to_item.to('cuda')


        elif init_way.lower() == 'oldembrqvae':
            self.tree_height = 4 # rqvqe layer
            num_all_leaf_code =  10 * self.k ** (self.tree_height - 1) #3 * 256 * 256 * 256  #self.k ** (self.tree_height - 1) +  # the number of leaf node
            self.code_to_item = torch.zeros((num_all_leaf_code,), dtype=torch.int64)
            self.item_to_code = {item_id: [] for item_id in range(self.item_num)}  # record the path
            print('start to construct for rqvae')


            index = torch.arange(self.item_num)
            # t5
            leaf_node_codes = torch.load('/home/wangye/wangye2/wangye/recommend/RecForest-main/lib/rqvae/mind_rqvae_code')#torch.randint (0, 256, (self.item_num, 3))
            # # bert
            # leaf_node_codes = torch.load(
            #     '/home/wangye/wangye2/wangye/recommend/RecForest-main/bert_full_t5_rqvae5_code')
            for i, code in zip(range(len(index)), leaf_node_codes):
                real_item_id = index[i].item()
                reverse_path = []
                for j in range(self.tree_height):
                    reverse_path.append(((code[j]) % self.k).item())
                self.item_to_code[real_item_id].append(torch.LongTensor(reverse_path))
                unique_code = 0
                for j in range(self.tree_height - 1):
                    unique_code += (self.k ** j * code[self.tree_height-1-j-1].item())
                unique_code += self.k ** (self.tree_height - 1) * code[self.tree_height - 1].item()
                # if int(unique_code) > len(self.code_to_item):
                #     print(code)
                self.code_to_item[int(unique_code)] = real_item_id
            self.card = torch.zeros(self.tree_height)
            # for i in range(self.tree_height):
            #     self.card[i] = self.k ** (self.tree_height - i - 1)  # recover to the code
            for i in range(0, self.tree_height - 1):
                self.card[i] = self.k ** (self.tree_height - (i + 1) - 1)
            self.card[-1] = self.k ** (self.tree_height - 1)
            self.card = self.card.to('cuda')
            self.code_to_item = self.code_to_item.to('cuda')

        elif init_way.lower()=='datakm' or init_way.lower()=='embkm':
            print('start to construct')
            assert data is not None
            print(data.shape)
            print(self.item_num)
            assert data.shape[0]==self.item_num
            # print(data.shape[0])
            constructer=ConstructKmeansTree(parall=parall, device=self.device)
            if data.shape[0]<num_all_leaf_code:
                index=torch.cat([torch.arange(self.item_num),\
                    torch.randint(low=0,high=self.item_num,size=(num_all_leaf_code-self.item_num,))],dim=0)
                index[:]=index[torch.randperm(index.nelement())]
                assert len(index)==num_all_leaf_code
                #data=None,k=4,max_iters=100
                #print('start to construct')
                item_ids,leaf_node_codes=constructer.train(data[index],self.k,max_iters=max_iters,feature_ratio=feature_ratio)
                start_code,end_code=leaf_node_codes.min().item(),leaf_node_codes.max().item()
                assert start_code==(self.k**self.tree_height-1)/(self.k-1) and end_code==(self.k**(self.tree_height+1)-1)/(self.k-1)-1
                #print(len(index),len(item_ids),len(leaf_node_codes))
                for i,j,code in zip(range(len(index)),item_ids,leaf_node_codes):
                    assert i==j.item()
                    real_item_id=index[i].item()                    
                    reverse_path,tc=[],code.item()
                    assert tc>=start_code and tc<=end_code
                    for _ in range(self.tree_height):
                        reverse_path.append((tc-1)%self.k)
                        tc=(tc-1)//self.k
                    self.item_to_code[real_item_id].append(torch.LongTensor(reverse_path[::-1]))
                    self.code_to_item[code.item()-start_code]=real_item_id
            self.card = torch.zeros(self.tree_height)
            for i in range(self.tree_height):
                self.card[i] = self.k ** (self.tree_height - i - 1)# recover to the code
            self.card = self.card.to('cuda')
            self.code_to_item = self.code_to_item.to('cuda')
    
    def read_tree(self, code_to_item_file, item_to_code_file,k=4):

        self.code_to_item = torch.tensor(np.load(code_to_item_file)).to('cuda')

        self.item_num=self.code_to_item.max().item()+1
        #self.item_to_code={item_id:[] for item_id in range(self.item_num)}
        item_to_code_mat = torch.tensor(np.load(item_to_code_file)).long()
        assert self.item_num==item_to_code_mat.shape[0]
        self.tree_height = item_to_code_mat.shape[-1]

        # for item_id,row in enumerate(item_to_code_mat):
        #     self.item_to_code[item_id].append(row)
        self.item_to_code = item_to_code_mat.to('cuda')
        self.k=k
        self.card = torch.zeros(self.tree_height).to('cuda')
        for i in range(self.tree_height):
            self.card[i] = self.k ** (self.tree_height - i - 1)# recover to the code 
        return self
    
    def label_to_path(self, batch_y):
        """
        batch_y is[bs,1], it contains item ids
        given the item idd, obtain the path sequence
        """
        #print(self.item_to_code1[batch_y].shape)
        #return torch.LongTensor(self.item_to_code[batch_y].numpy())
        # return  torch.cat([random.sample(self.item_to_code[label],1)[0] \
        #     for label in batch_y.view(-1).numpy()],dim=-1).view(len(batch_y),1,self.tree_height)
        return  torch.cat([self.item_to_code[label][0] \
            for label in batch_y.view(-1).numpy()],dim=-1).view(len(batch_y),1,self.tree_height)


    
    def path_to_label(self, batch_pred_seq):#translate the path sequence to item
        """
        batch_pred_seq [batch_size,topk,tree_height]
        """

        #print(card)
        batch_pred_seq, self.card = batch_pred_seq.to(self.device) , self.card.to(self.device)
        batch_code_value = ((batch_pred_seq * self.card).sum(-1)).long()
        #-----------------
        # count = 0
        # for i in batch_code_value:
        #     for j in i:
        #         if int(j) in set(list(self.code_to_item.keys())):
        #             count +=1
        # print(count,batch_code_value.shape[0]*batch_code_value.shape[1])
        #-----------------
        return self.code_to_item[batch_code_value]


