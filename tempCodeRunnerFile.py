import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

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
soup = BeautifulSoup(content, 'html.parser')
s_tags = soup.find_all('s')

# Nhóm các câu theo docid
docid_groups = {}
for s in s_tags:
    docid = s['docid']
    if docid not in docid_groups:
        docid_groups[docid] = []
    docid_groups[docid].append(s)
print('****************docid***************')
print(docid_groups)
# Tóm tắt văn bản cho từng docid và tạo các thẻ <s> mới
new_s_tags = []
for docid, sentences in docid_groups.items():
    # Ghép các câu lại thành một văn bản duy nhất
    text = ' '.join([s.get_text() for s in sentences])
    print('***************text***************')
    print(text)
    # Tóm tắt văn bản
    summary = textrank_summarizer(text)
    
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
output_file_path = 'd061j_output_new'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(new_content)

# In ra nội dung mới
print(new_content)