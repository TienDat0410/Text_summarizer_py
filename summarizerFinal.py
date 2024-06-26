import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter, defaultdict
import math
import nltk
from bs4 import BeautifulSoup

nltk.download('punkt')

def compute_tf(sentences):
    tf = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        word_count = Counter(words)
        total_words = len(words)
        tf.append({word: count / total_words for word, count in word_count.items()})
        print('tfidf', tf)
    return tf

def compute_idf(sentences):
    idf = defaultdict(lambda: 0)
    total_sentences = len(sentences)
    for sentence in sentences:
        words = set(word_tokenize(sentence.lower()))
        for word in words:
            idf[word] += 1
    for word, count in idf.items():
        idf[word] = math.log(total_sentences / (1 + count))
    return idf

def compute_tfidf(tf, idf):
    tfidf = []
    for sentence_tf in tf:
        sentence_tfidf = {word: tf_value * idf[word] for word, tf_value in sentence_tf.items()}
        tfidf.append(sentence_tfidf)
    return tfidf

def cosine_similarity(vec1, vec2):
    common_words = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    norm1 = math.sqrt(sum(value**2 for value in vec1.values()))
    norm2 = math.sqrt(sum(value**2 for value in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def build_similarity_matrix(sentences, tfidf):
    size = len(sentences)
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(tfidf[i], tfidf[j])
    return similarity_matrix

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

def textrank_summarizer(text, num_sentences):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return ""

    # Tính TF, IDF, và TF-IDF
    tf = compute_tf(sentences)
    idf = compute_idf(sentences)
    tfidf = compute_tfidf(tf, idf)

    # Tính độ tương đồng cosine giữa các câu
    sim_matrix = build_similarity_matrix(sentences, tfidf)

    # Tính điểm PageRank
    scores = pagerank(sim_matrix)

    # Xếp hạng các câu theo điểm số
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]

    return ' '.join(summary_sentences)

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
    # num_sentences = int(sentences[0]['num'])
    num_sentences = int((len(sentences) *10) / 100)
    
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
