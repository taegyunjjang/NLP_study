import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVD
np.set_printoptions(precision=3)
U, S, V = np.linalg.svd(W)
# print(f'C[0] = {C[0]}')
# print(f'W = {W}')
# print(f'U = {U}')
print(f'S = {S}')
# print(f'V = {V}')
# print(f'orthogonal check')
# print(f'U[:,0] = {U[:,0]}')
# print(f'U[:,1] = {U[:,1]}')
# print(f'내적 값 = {np.dot(V[0], V[1])}')

'''
os나 넘파이 버전 등 환경에 따라 값이나 축의 순서가 다를 수 있음
차원 감소 목적으로의 효과는 동일
'''
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
