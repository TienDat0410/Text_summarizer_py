import re
import math
from collections import defaultdict
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import numpy as np
from numpy.linalg import norm
# 
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')


# Tokenize câu
# def sentence_tokenize(text):
#     return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

# Tokenize từ
# def word_tokenize(sentence):
#     return re.findall(r'\w+', sentence.lower())

# Tính độ tương đồng cosine
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    # print('keys_vec1:',vec1.keys(), '\n');
    # print('value_vec1:',[vec1[x]for x in intersection], '\n');
    # print('keys_vec2:',vec2.keys(), '\n');
    # print('value_vec2:',[vec2[x] for x in intersection], '\n');
    if not denominator:
        return 0.01
    else:
        # print('cosine_similarity:',float(numerator) / denominator, '\n');
        return float(numerator) / denominator


# Tính TF-IDF cho từng câu
def compute_tfidf(sentences):
    word_freq = defaultdict(int)
    for sentence in sentences:
        words = set(word_tokenize(sentence))
        for word in words:
            word_freq[word] += 1

    tfidf = []
    for sentence in sentences:
        tfidf_sentence = {}
        words = word_tokenize(sentence)
        word_count = Counter(words)
        for word in words:
            tf = word_count[word] / len(words)
            idf = math.log(len(sentences) / (word_freq[word]))
            tfidf_sentence[word] = tf * idf
        tfidf.append(tfidf_sentence)
        print(tfidf)

    return tfidf

# Tính ma trận độ tương đồng
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

# Tính điểm PageRank
def pagerank(similarity_matrix, eps=0.0001, d=0.85):
    size = len(similarity_matrix)
    rank = [1.0 / size] * size
    new_rank = [0] * size
    change = 1
    while change > eps:
        for i in range(size):
            new_rank[i] = (1 - d) / size + d * sum(similarity_matrix[j][i] * rank[j] for j in range(size))
        change = sum(abs(new_rank[i] - rank[i]) for i in range(size))
        rank = new_rank[:]
    return rank

# Chức năng tóm tắt văn bản
def textrank_summarizer(text, num_sentences):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return ""

    tfidf = compute_tfidf(sentences)
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(sentences)
    # Tính độ tương đồng cosine 
    # sim_matrix = cosine_similarity(X, X)
    similarity_matrix = build_similarity_matrix(sentences, tfidf)
    scores = pagerank(similarity_matrix)
    # similarity_matrix = build_similarity_matrix(sentences, tfidf)
    # scores = pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]

    return ' '.join(summary_sentences)
#************************************************************************************************************
# Đọc tệp đầu vào và phân tích cú pháp XML
# with open('d061j', 'r', encoding='utf-8') as file:
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

    # Tóm tắt văn bản, sử dụng số lượng câu từ giá trị num của câu đầu tiên
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
# output_file_path = 'd061j_output_new.xml'
output_file_path = 'd070f_text_output'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_content)

# In ra nội dung mới
# print(new_content)