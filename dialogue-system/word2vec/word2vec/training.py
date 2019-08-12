# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
#配置再训练时log的格式了
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word2vec():
    corpus = open('./data/test_seg.txt', 'r', encoding='utf-8')   # 1. 读取训练语料
    model = Word2Vec(LineSentence(corpus), sg=0,size=192, window=5, min_count=5, workers=9)
    # 2. 一步进行词向量的训练，再简单不过了。这几个参数的意义，大家想一想
    model.save('train.word2vec')
    #3. 训练完之后把模型保存起来

if __name__ == '__main__':
    train_word2vec()
