import numpy as np
import w2v_utils

#计算两个词的相似度
def cosine_similarity(u,v):
    distance = 0

    dot = np.dot(u,v)
    norm_u = np.sqrt(np.sum(np.power(u,2)))
    norm_v = np.sqrt(np.sum(np.power(v,2)))

    cosine_similarity = np.divide(dot,norm_u*norm_v)

    return cosine_similarity

# 解决“A与B相比就类似于C与____相比一样”之类的问题
def compute_analogy(word_a,word_b,word_c,word_to_vec_map):
    #转小写
    word_a,word_b,word_c = word_a.lower(),word_b.lower(),word_c.lower()

    #获取词向量
    e_a,e_b,e_c = word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]

    #获取全部单词
    words = word_to_vec_map.key()

    #max_cosine_sim初始化为一个比较大的负数
    max_cosine_sim = -100
    best_word = None

    for word in words:
        # 要避免匹配到输入的数据
        if word in [word_a,word_b,word_c]:
            continue
        #计算余弦相似度
        cosine_sim = cosine_similarity((e_b-e_a),(word_to_vec_map[word]-e_c))

        if cosine_sim>max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word
        
    return best_word

if __name__ == "__main__":
    words,word_to_vec_map = w2v_utils.read_glove_vecs("D:/deep learning course/Course5/week2/data/glove.6B.50d.txt")

    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
