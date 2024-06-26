import re
import math
from collections import defaultdict
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')

# Tính điểm PageRank
# def pagerank(similarity_matrix, eps=0.0001, d=0.85):
#     size = len(similarity_matrix)
#     rank = [1.0 / size] * size
#     new_rank = [0] * size
#     change = 1
#     while change > eps:
#         for i in range(size):
#             new_rank[i] = (1 - d) / size + d * sum(similarity_matrix[j][i] * rank[j] for j in range(size))
#         change = sum(abs(new_rank[i] - rank[i]) for i in range(size))
#         rank = new_rank[:]
#     return rank
def pagerank(similarity_matrix, eps=0.0001, d=0.85):
    size = len(similarity_matrix)
    rank = np.ones(size) / size
    new_rank = np.zeros(size)
    change = 1

    while change > eps:
        for u in range(size):
            sum_rank = 0
            for v in range(size):
                if similarity_matrix[v][u] > 0:
                    sum_rank += rank[v] / np.sum(similarity_matrix[v])
            new_rank[u] = (1 - d) / size + d * sum_rank
        
        change = np.sum(np.abs(new_rank - rank))
        rank = new_rank.copy()
    
    return rank
# 
def build_similarity_matrix(sentences, tfidf):
    similarity_matrix = [[0 for _ in range(len(sentences))] for _ in range(len(sentences))]
    print('lenght_sentences:',len(sentences))
    for i in range(len(sentences)):
        row = []
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(tfidf[i], tfidf[j])
                row.append(f"{similarity_matrix[i][j]}")
            else:
                row.append("0.00")
            # print(", ".join(row))
    #print('similarity_matrix:',similarity_matrix, '\n');
    return similarity_matrix
#
def textrank_summarizer(text, num_sentences):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return ""

    # Tính TF-IDF cho các câu
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(sentences)
    print(X)

    # Tính độ tương đồng cosine giữa các câu
    sim_matrix = cosine_similarity(X, X)
    # print('matrix: \n', sim_matrix)

    scores = pagerank(sim_matrix)

    # Xếp hạng các câu theo điểm số
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]

    return ' '.join(summary_sentences)

#************************************************************************************************************
# Đọc tệp đầu vào và phân tích cú pháp XML
with open('d070f', 'r', encoding='utf-8') as file:
    content = file.read()

# Sử dụng BeautifulSoup để lọc dữ liệu từ các thẻ <s>
soup = BeautifulSoup(content, 'html.parser')
s_tags = soup.find_all('s')

# Nhóm các câu theo docid
docid_groups = {}
for s in s_tags:
    docid = s['docid']
    if docid not in docid_groups:
        docid_groups[docid] = []
    docid_groups[docid].append(s)

# Tóm tắt văn bản cho từng docid và tạo các thẻ <s> mới
new_s_tags = []
for docid, sentences in docid_groups.items():
    # Ghép các câu lại thành một văn bản duy nhất
    text = ' '.join([s.get_text() for s in sentences])

    # 
    num_sentences = int(sentences[0]['num'])
    
    summary = textrank_summarizer(text, num_sentences=num_sentences)

    # Chia bản tóm tắt thành các câu
    summary_sentences = sent_tokenize(summary)

    # Tạo các thẻ <s> mới với thuộc tính docid, num và wdcount từ câu ban đầu
    for summary_sentence in summary_sentences:
        for s in sentences:
            if summary_sentence in s.get_text():
                num = s['num']
                wdcount = len(word_tokenize(summary_sentence))
                new_s_tag = soup.new_tag('s', docid=docid, num=num, wdcount=wdcount)
                new_s_tag.string = summary_sentence
                new_s_tags.append(new_s_tag)
                break

# Tạo văn bản mới với các thẻ <s> mới
new_content = '\n'.join(str(tag) for tag in new_s_tags)

# Lưu văn bản mới vào tệp
output_file_path = 'd070f_text_output'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_content)

# In ra nội dung mới
print('success summarizer')

print(new_content)