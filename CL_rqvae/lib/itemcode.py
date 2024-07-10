import torch

class Tree:
    def __init__(self, codes, k, item_num):
        self.codes = codes
        self.k = k
        self.item_num = item_num

        self.tree_height = self.codes.shape[1]
        num_all_leaf_code =  self.tree_height * self.k ** (self.tree_height) #3 * 256 * 256 * 256  #self.k ** (self.tree_height - 1) +  # the number of leaf node
        print("num_all_leaf_code:",num_all_leaf_code)
        self.code_to_item = torch.zeros((num_all_leaf_code,), dtype=torch.int64)
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
        for i in range(0, self.tree_height):
            self.card[i] = self.k ** (self.tree_height - (i + 1))
    