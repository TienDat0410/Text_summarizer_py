import re
import math
from collections import defaultdict, Counter

# Tokenize câu
def sentence_tokenize(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

# Tokenize từ
def word_tokenize(sentence):
    return re.findall(r'\w+', sentence.lower())

# Tính độ tương đồng cosine
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
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
    
    return tfidf

# Tính ma trận độ tương đồng
def build_similarity_matrix(sentences, tfidf):
    similarity_matrix = [[0 for _ in range(len(sentences))] for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(tfidf[i], tfidf[j])
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
    sentences = sentence_tokenize(text)
    if len(sentences) == 0:
        return ""
    
    tfidf = compute_tfidf(sentences)
    similarity_matrix = build_similarity_matrix(sentences, tfidf)
    scores = pagerank(similarity_matrix)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]
    
    return ' '.join(summary_sentences)


text = """
heat peaked in the last two days, with the maximum temperature exceeding 39 degrees on Thursday.
These temperatures were measured at meteorological offices, with actual outdoor temperatures being two to four degrees higher, and up to 10 degrees on concrete roads and buildings with metal roofs.
Central Vietnam has experienced a heat wave since June 9 with temperatures of 39-40 degrees in Do Luong District of Nghe An Province and Huong Khe District in Ha Tinh Province.
"""

# Sử dụng hàm tóm tắt văn bản để tóm tắt văn bản đầu vào
summary = textrank_summarizer(text, num_sentences=5)
output_file_path = 'output_new'
# output_file_path = 'text_output'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(summary)
print("Tóm tắt văn bản:")
print(summary)