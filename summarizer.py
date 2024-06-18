import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Tải các gói dữ liệu cần thiết
nltk.download('punkt')

def textrank_summarizer(text, num_sentences=3):
    # Chia văn bản thành các câu
    sentences = sent_tokenize(text)
    
    # Tính TF-IDF cho từng câu
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    # Tính độ tương đồng cosine giữa các câu
    sim_matrix = cosine_similarity(X, X)
    
    # Tạo đồ thị từ ma trận độ tương đồng
    nx_graph = nx.from_numpy_array(sim_matrix)
    
    # Tính điểm PageRank cho từng câu
    scores = nx.pagerank(nx_graph)
    
    # Sắp xếp các câu theo điểm số PageRank
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Chọn các câu có điểm cao nhất
    summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]
    
    # Tạo bản tóm tắt
    summary = ' '.join(summary_sentences)
    
    return summary

# Đọc tệp đầu vào và phân tích cú pháp XML
with open('d061j', 'r', encoding='utf-8') as file:
    content = file.read()

# Sử dụng BeautifulSoup để lọc dữ liệu từ các thẻ <s>
soup = BeautifulSoup(content, 'lxml')
s_tags = soup.find_all('s')

# Lấy thông tin và nội dung từ các thẻ <s>
sentences = []
info = []
for s in s_tags:
    sentences.append(s.get_text())
    info.append({
        'docid': s['docid'],
        'num': s['num'],
        'wdcount': s['wdcount']
    })

# Ghép tất cả các câu lại thành một văn bản duy nhất
text = ' '.join(sentences)

# Tóm tắt văn bản
summary = textrank_summarizer(text)

# Chia bản tóm tắt thành các câu
summary_sentences = sent_tokenize(summary)

# Tạo các thẻ <s> mới với thuộc tính docid, num và wdcount từ câu ban đầu
new_s_tags = []
for summary_sentence in summary_sentences:
    for i, sentence in enumerate(sentences):
        if summary_sentence in sentence:
            num = info[i]['num']
            wdcount = len(word_tokenize(summary_sentence))
            docid = info[i]['docid']
            new_s_tag = soup.new_tag('s', docid=docid, num=num, wdcount=wdcount)
            new_s_tag.string = summary_sentence
            new_s_tags.append(new_s_tag)
            break

# Tạo văn bản mới với các thẻ <s> mới
new_content = '\n'.join(str(tag) for tag in new_s_tags)

# Lưu văn bản mới vào tệp
output_file_path = 'd061j_output_new.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_content)

# In ra nội dung mới
print(new_content)