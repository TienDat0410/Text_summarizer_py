from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import numpy as np

nltk.download('punkt')

def textrank_summarizer(text, num_sentences):
    sentences = sent_tokenize(text)
    
    if len(sentences) == 0:
        return ""
    
    # Tạo đồ thị từ ma trận độ tương đồng
    sim_matrix = create_similarity_matrix(sentences)
    
    # In ra các nodes và edges
    print_graph(sim_matrix, sentences)
    
    # Tính điểm PageRank 
    scores = pagerank(sim_matrix, eps=0.0001, d=0.85)
    
    # Sắp xếp theo PageRank
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    summary_sentences = [s for scores, s in ranked_sentences[:num_sentences]]
    
    # Tạo bản tóm tắt sau khi tính
    summary = ' '.join(summary_sentences)
    
    return summary

def create_similarity_matrix(sentences):
    size = len(sentences)
    sim_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                sim_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return sim_matrix

def sentence_similarity(sent1, sent2):
    words1 = set(word_tokenize(sent1))
    words2 = set(word_tokenize(sent2))
    all_words = words1.union(words2)
    
    vec1 = [1 if word in words1 else 0 for word in all_words]
    vec2 = [1 if word in words2 else 0 for word in all_words]
    
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def print_graph(similarity_matrix, sentences):
    print("Nodes:")
    for i, sentence in enumerate(sentences):
        print(f"Node {i}: {sentence}")
    
    print("\nEdges:")
    size = len(similarity_matrix)
    for i in range(size):
        for j in range(size):
            if i != j and similarity_matrix[i][j] > 0:
                print(f"Edge from Node {i} to Node {j} with weight {similarity_matrix[i][j]:.4f}")

# Tính điểm PageRank
def pagerank(similarity_matrix, eps=0.0001, d=0.85):
    size = len(similarity_matrix)
    rank = [1.0 / size] * size
    new_rank = [0] * size
    change = 1
    
    # Khi ngưỡng thay đổi bé hơn eps thì dừng
    while change > eps:
        for i in range(size):
            new_rank[i] = (1 - d) / size + d * sum(similarity_matrix[j][i] * rank[j] for j in range(size))
        change = sum(abs(new_rank[i] - rank[i]) for i in range(size))
        rank = new_rank[:]
    return rank

# Đọc tệp đầu vào.
with open('d061j', 'r', encoding='utf-8') as file:
    content = file.read()

# Sử dụng BeautifulSoup để lọc thẻ <s>
soup = BeautifulSoup(content, 'html.parser')
s_tags = soup.find_all('s')

# Nhóm theo docid
docid_groups = {}
for s in s_tags:
    docid = s['docid']
    if docid not in docid_groups:
        docid_groups[docid] = []
    docid_groups[docid].append(s)

# Lấy nội dung chính không lấy các thông tin như docID,..
new_s_tags = []
for docid, sentences in docid_groups.items():
    # Ghép các câu
    text = ' '.join([s.get_text() for s in sentences])
    
    # Lấy 10%
    num_sentences = int((len(sentences) * 10) / 100)
    
    summary = textrank_summarizer(text, num_sentences=num_sentences)
    
    # Chia V bản tóm tắt thành các câu
    summary_sentences = sent_tokenize(summary)
    
    # Tạo các thẻ <s> mới
    for summary_sentence in summary_sentences:
        for s in sentences:
            if summary_sentence in s.get_text():
                num = s['num']
                wdcount = len(word_tokenize(summary_sentence))
                new_s_tag = soup.new_tag('s', docid=docid, num=num, wdcount=wdcount)
                new_s_tag.string = summary_sentence
                new_s_tags.append(new_s_tag)
                break

# Tạo văn bản mới 
new_content = '\n'.join(str(tag) for tag in new_s_tags)

# Lưu văn bản mới vào tệp
output_file_path = '061_text_output'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_content)

# In ra nội dung mới
# print(new_content)
print('------------------------success summarizer---------------------------')