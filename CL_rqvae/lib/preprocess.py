from collections import UserDict
import pandas as pd
import random
import numpy as np

# data_set_name = 'MIND'
# seq = 70
# minseq = 15


def preprocess_data_mind(data_set_name, seq, minseq):
    raw_inter_file = 'data/{}/{}.inter'.format(data_set_name, data_set_name) #交互文件
    raw_item_file = 'data/{}/news.tsv'.format(data_set_name) # 描述文件
    tsv_file = pd.read_csv(raw_item_file, sep='\t', header = None, keep_default_na = False)
    id = tsv_file.values[:, 0]
    id_list = []
    for i in id:
        id_list.append(int(i[1:]))
    id_set= set(id_list) #描述文件的全部Item id

    print("----------------------filter掉不符合要求的user-------------------------")

    user_history = {}
    with open(raw_inter_file ,'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            arr = line.split('\t')
            if int(arr[2]) == 1.0:
                user_id = int(arr[0])
                item_id = int(arr[1])
                if user_id in user_history.keys():
                    curr = user_history[user_id]
                    curr.append(item_id)
                    user_history[user_id] = curr
                else:
                    user_history.update({user_id:[item_id]})

    # print(len(user_history))

    new_user_history = {}
    for k, v in user_history.items():
        if len(v)>=minseq+2:
            new_user_history.update({k: v})

    user_list = []
    item_list = []

    for k, v in new_user_history.items():
        user_list.append(k)
        for i in v:
            item_list.append(i)

    user_set = set(user_list)
    item_set = set(item_list)

    final_itemset = item_set.intersection(id_set)
    print(len(id_set), len(user_list), len(user_set), len(item_list), len(item_set), len(final_itemset))
    # 33692 33692 820626 12646

    user_list = list(user_set)
    item_list = list(final_itemset)
    print(len(user_list), len(item_list))



    print("----------------------------按照时间戳排序，卡长度，统计样本数据-----------------------------")
    user_his_behav = dict()
    with open(raw_inter_file ,'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            arr = line.split('\t')
            curr_user = int(arr[0])
            curr_item = int(arr[1])
            curr_label = int(arr[2])
            curr_time = int(arr[2])
            if curr_label == 1.0 and curr_user in user_set and curr_item in final_itemset:
                if curr_user not in user_his_behav.keys():
                    user_his_behav[curr_user] = list()
                user_his_behav[curr_user].append((curr_item, curr_time))

    for _, value in user_his_behav.items():
        value.sort(key=lambda x: x[1])

    # -----卡长度-----
    maxn = 0
    minn = 10000
    mean = 0.0
    dict_ = {}
    count = 0.0
    # print(user_his_behav)
    print(len(user_his_behav))
    new_user_his_behav = dict()
    for k,v in user_his_behav.items():
        x = len(v)
        if x>=minseq+2 and x<=seq:
            maxn = max(maxn, x)
            minn = min(minn, x)
            mean = mean + x
            count += 1
            if not x in dict_.keys():
                dict_[x] = 0
            dict_[x] = dict_[x] + 1

            curr = []
            for i in v:
                curr.append(i[0])
            new_user_his_behav.update({k:curr})

    mean = mean / count

    sorted_len_dict = sorted(dict_.items(), key=lambda x:x[1])
    totalcount = sum(dict_.values())
    targetvalue = totalcount/2
    currentcount = 0
    for k, v in sorted_len_dict:
        currentcount +=v
        if currentcount>=targetvalue:
            # print(k)
            break
    print('max{}, min{}, mean{}, mid{}, len{}'.format(maxn, minn, mean, k, len(dict_)))
    user_his_behav = new_user_his_behav
    print(len(user_his_behav))

    print("------------------------------存储字典User和item----------------------------------")

    user_list = []
    item_list = []
    for k, v in user_his_behav.items():
        user_list.append(k)
        for i in v:
            item_list.append(i)

    user_set = set(user_list)
    item_set = set(item_list)
    final_itemset = item_set.intersection(id_set)
    user_list = list(user_set)
    item_list = list(final_itemset)

    user_ids={id:i for i,id in enumerate(user_list)}
    item_ids_new={id:i for i,id in enumerate(item_list)}
    print('user number {}, item number {}'.format(len(user_ids),len(item_ids_new)))
    item_num_node_num_file='data/{}/item_node_num.txt'.format(data_set_name)
    with open(item_num_node_num_file, 'w') as file:
        file.write(str(len(user_ids)))
        file.write("\n")
        file.write(str(len(item_ids_new)))

    item_dict_path = 'data/{}/item_dict'.format(data_set_name)
    np.save(str(item_dict_path), item_ids_new)

    print("-------------------------划分userID为训练验证和测试--------------------")
    user_list_len = len(user_list)
    random.shuffle(user_list)
    train_len, val_len = int(0.8 * user_list_len), int(0.1 * user_list_len)

    # train_user_id = user_list[:train_len]
    # val_user_id = user_list[train_len:train_len+val_len]
    # test_user_id = user_list[train_len+val_len:]

    train_instances_file='data/{}/train_instances'.format(data_set_name)
    val_instances_file='data/{}/val_instances'.format(data_set_name)
    test_instances_file='data/{}/test_instances'.format(data_set_name)

    cnt = -1
    # itobj =[train_instances_file, val_instances_file, test_instances_file]# , [user_list,user_list,user_list]) #[train_user_id, val_user_id, test_user_id])
    with open(train_instances_file, 'w') as f1, open(val_instances_file, 'w') as f2, open(test_instances_file, 'w') as f3:
        count0, count1, count2 = 0,0,0
        for user, history in user_his_behav.items():
            currcount = len(history)
            if currcount < minseq + 2:
                continue
            arr = [-1 for i in range(seq - minseq)] + [item_ids_new[v] for v in history]
            for i in range(len(arr) - seq - 1):
                sample = arr[i: i + seq]
                count0 +=1
                f1.write('{}|'.format(user_ids[user]))  # sample id
                f1.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                f1.write("{}".format(sample[-1]))  # label, no ts
                f1.write('\n')

            i = len(arr) - seq - 1
            count1+=1
            sample = arr[i: i + seq]
            f2.write('{}|'.format(user_ids[user]))  # sample id
            f2.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
            f2.write("{}".format(sample[-1]))  # label, no ts
            f2.write('\n')

            i = len(arr) - seq
            count2+=1
            sample = arr[i: i + seq]
            f3.write('{}|'.format(user_ids[user]))  # sample id
            f3.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
            f3.write("{}".format(sample[-1]))  # label, no ts
            f3.write('\n')

        print(count0, count1, count2)



def preprocess_data_amazon(data_set_name, seq, minseq, ratingbar):
    raw_inter_file = 'data/{}/{}.inter'.format(data_set_name, data_set_name) #交互文件
    raw_item_file = 'data/{}/{}.item'.format(data_set_name, data_set_name) # 描述文件
    tsv_file = pd.read_csv(raw_item_file, sep='\t', header = None, keep_default_na = False)
    id = tsv_file.values[1:, 0]
    id_set= set(id) #描述文件的全部Item id

    print("----------------------filter掉不符合要求的user-------------------------")

    user_history = {}
    with open(raw_inter_file ,'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                print(line)
                continue
            arr = line.split('\t')
            if float(arr[2]) >= ratingbar:
                user_id = arr[0]
                item_id = arr[1]
                if user_id in user_history.keys():
                    curr = user_history[user_id]
                    curr.append(item_id)
                    user_history[user_id] = curr
                else:
                    user_history.update({user_id:[item_id]})

    # print(len(user_history))

    new_user_history = {}
    for k, v in user_history.items():
        if len(v)>=minseq+2:
            new_user_history.update({k: v})

    user_list = []
    item_list = []

    for k, v in new_user_history.items():
        user_list.append(k)
        for i in v:
            item_list.append(i)

    user_set = set(user_list)
    item_set = set(item_list)
    final_itemset = item_set.intersection(id_set)

    print(len(id_set), len(user_list), len(user_set), len(item_list), len(item_set), len(final_itemset))
    # 33692 33692 820626 12646

    user_list = list(user_set)
    item_list = list(final_itemset)
    print(len(user_list), len(item_list))

    print("----------------------------按照时间戳排序，卡长度，统计样本数据-----------------------------")
    user_his_behav = dict()
    with open(raw_inter_file ,'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            arr = line.split('\t')
            curr_user = str(arr[0])
            curr_item = str(arr[1])
            curr_label = float(arr[2])
            curr_time = float(arr[2])
            if curr_label >= ratingbar and curr_user in user_set and curr_item in final_itemset:
                if curr_user not in user_his_behav.keys():
                    user_his_behav[curr_user] = list()
                user_his_behav[curr_user].append((curr_item, curr_time))

    for _, value in user_his_behav.items():
        value.sort(key=lambda x: x[1])

    # -----卡长度-----
    maxn = 0
    minn = 10000
    mean = 0.0
    dict_ = {}
    count = 0.0
    # print(user_his_behav)
    print(len(user_his_behav))
    new_user_his_behav = dict()
    for k,v in user_his_behav.items():
        x = len(v)
        if x>=minseq+2 and x<=seq:
            maxn = max(maxn, x)
            minn = min(minn, x)
            mean = mean + x
            count += 1
            if not x in dict_.keys():
                dict_[x] = 0
            dict_[x] = dict_[x] + 1

            curr = []
            for i in v:
                curr.append(i[0])
            new_user_his_behav.update({k:curr})

    mean = mean / count

    sorted_len_dict = sorted(dict_.items(), key=lambda x:x[1])
    totalcount = sum(dict_.values())
    targetvalue = totalcount/2
    currentcount = 0
    for k, v in sorted_len_dict:
        currentcount +=v
        if currentcount>=targetvalue:
            # print(k)
            break
    print('max{}, min{}, mean{}, mid{}, len{}'.format(maxn, minn, mean, k, len(dict_)))
    user_his_behav = new_user_his_behav
    print(len(user_his_behav))

    print("------------------------------存储字典User和item----------------------------------")

    user_list = []
    item_list = []
    for k, v in user_his_behav.items():
        user_list.append(k)
        for i in v:
            item_list.append(i)

    user_set = set(user_list)
    item_set = set(item_list)
    final_itemset = item_set.intersection(id_set)
    user_list = list(user_set)
    item_list = list(final_itemset)

    user_ids={id:i for i,id in enumerate(user_list)}
    item_ids_new={id:i for i,id in enumerate(item_list)}
    # print(item_ids_new['B000002AU0'])

    print('user number {}, item number {}'.format(len(user_ids),len(item_ids_new)))
    item_num_node_num_file='data/{}/item_node_num.txt'.format(data_set_name)
    with open(item_num_node_num_file, 'w') as file:
        file.write(str(len(user_ids)))
        file.write("\n")
        file.write(str(len(item_ids_new)))

    item_dict_path = 'data/{}/item_dict'.format(data_set_name)
    np.save(str(item_dict_path), item_ids_new)

    print("-------------------------划分userID为训练验证和测试--------------------")
    user_list_len = len(user_list)
    random.shuffle(user_list)
    train_len, val_len = int(0.8 * user_list_len), int(0.1 * user_list_len)

    # train_user_id = user_list[:train_len]
    # val_user_id = user_list[train_len:train_len+val_len]
    # test_user_id = user_list[train_len+val_len:]

    train_instances_file='data/{}/train_instances'.format(data_set_name)
    val_instances_file='data/{}/val_instances'.format(data_set_name)
    test_instances_file='data/{}/test_instances'.format(data_set_name)

    cnt = -1
    with open(train_instances_file, 'w') as f1, open(val_instances_file, 'w') as f2, open(test_instances_file, 'w') as f3:
        count0, count1, count2 = 0,0,0
        for user, history in user_his_behav.items():
            currcount = len(history)
            if currcount < minseq + 2:
                continue
            arr = [-1 for i in range(seq - minseq)] + [item_ids_new[v] for v in history]
            for i in range(len(arr) - seq - 1):
                sample = arr[i: i + seq]
                count0 +=1
                f1.write('{}|'.format(user_ids[user]))  # sample id
                f1.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                f1.write("{}".format(sample[-1]))  # label, no ts
                f1.write('\n')

            i = len(arr) - seq - 1
            count1+=1
            sample = arr[i: i + seq]
            f2.write('{}|'.format(user_ids[user]))  # sample id
            f2.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
            f2.write("{}".format(sample[-1]))  # label, no ts
            f2.write('\n')

            i = len(arr) - seq
            count2+=1
            sample = arr[i: i + seq]
            f3.write('{}|'.format(user_ids[user]))  # sample id
            f3.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
            f3.write("{}".format(sample[-1]))  # label, no ts
            f3.write('\n')

        print(count0, count1, count2)



# for filename in itobj:
#     cnt = cnt+1
#     if cnt == 0:
#         train_len = 0
#         with open(filename, 'w') as f:
#             for user, history in user_his_behav.items():
#                 if user in UserIDlist:
#                     currcount = len(history)
#                     if currcount < minseq + 2:
#                         continue
#                     arr = [-1 for i in range(seq - minseq)] + [v for v in history]
#                     i = len(arr) - seq - 2
#                     sample = arr[i: i + seq]
#                     f.write('{}|'.format(user))  # sample id
#                     f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
#                     f.write("{}".format(sample[-1]))  # label, no ts
#                     f.write('\n')
#             print("写入训练文件")
#         print("train_len",train_len)

#     elif cnt == 1:
#         val_len = 0
#         with open(filename, 'w') as f:
#             for user, history in user_his_behav.items():
#                 if user in UserIDlist:
#                     currcount = len(history)
#                     if currcount < minseq + 2:
#                         continue
#                     arr = [-1 for i in range(seq - minseq)] + [v for v in history]
#                     i = len(arr) - seq - 1
#                     sample = arr[i: i + seq]
#                     f.write('{}|'.format(user))  # sample id
#                     f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
#                     f.write("{}".format(sample[-1]))  # label, no ts
#                     f.write('\n')
#             print("写入验证文件")
#         print("val_len",val_len)

#     elif cnt == 2:
#         test_len = 0
#         with open(filename, 'w') as f:
#             for user, history in user_his_behav.items():
#                 if user in UserIDlist:
#                     currcount = len(history)
#                     if currcount < minseq + 2:
#                         continue
#                     arr = [-1 for i in range(seq - minseq)] + [v for v in history]
#                     i = len(arr) - seq
#                     sample = arr[i: i + seq]
#                     f.write('{}|'.format(user))  # sample id
#                     f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
#                     f.write("{}".format(sample[-1]))  # label, no ts
#                     f.write('\n')
#             print("写入测试文件")
#         print("test_len",test_len)


