/data文件夹下放数据, /data/MIND/MIND.inter和news.tsv分别是user item交互文件和item文本描述文件, 下载链接https://drive.google.com/drive/folders/1MucMieAUkjbAZVka3mxTaGuGvXbA1SYT

/pretrain文件夹下面放bert-base-uncased和sentence-t5-base的本地文件，从hugging face下载

数据和模型参数放好后，运行main.py即可

数据预处理lib/preprocess.py
提取特征用lib/new_emb.py
训练rqvae用lib/train_rqvae.py

参数说明：
have_processed_data，是否预处理过数据， False为没处理过
have_processed_feat，是否进行过text -> embed的特征提取，False为没提取过
have_processed_code，是否feat->code
tree_has_generated = False # 是否已经构建code-to-item和item-to-code文件
k在rqvae训练时候和codebook_size需要保持一致；在random和embkm时候根据树的分支个数自行设置

# cost
