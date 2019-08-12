#coding=utf-8
import gensim
def count_similarity():

    # 1 当然时加载词向量了
    model = gensim.models.Word2Vec.load('./train.word2vec')
    #2. 计算两个词的相似度
    print(model.similarity('数学','天文学'))  #相似度为0.93
    print(model.similarity('数学','德国'))  #相似度为0.22

    word = '中国'
    # 查找词典中，与其最相近的词
    if word in model.wv.index2word:
        print(model.most_similar(word))  ## [('大陆', 0.7592229247093201), ('内地', 0.6802576780319214)
                                            # , ('台湾', 0.6557508111000061), ('中华民族', 0.648242175579071)。。。

if __name__ == '__main__':
    count_similarity()