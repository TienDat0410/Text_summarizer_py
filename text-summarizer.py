import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
import re

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Đọc nội dung từ file đầu vào
with open('d061j', 'r') as file:
    text = file.read()

# Tách câu từ nội dung, giả sử mỗi câu nằm trong thẻ <s>...</s>
sentences = re.findall(r'<s docid="(.*?)" num="(.*?)" wdcount="(.*?)">(.*?)</s>', text)

# Tạo từ điển để đếm tần suất từ
word_freq = defaultdict(int)
stop_words = set(stopwords.words('english') + list(string.punctuation))

# Đếm tần suất từ trong các câu
for _, _, _, sentence in sentences:
    words = word_tokenize(sentence.lower())
    for word in words:
        if word not in stop_words:
            word_freq[word] += 1

# Tính điểm cho mỗi câu dựa trên tần suất từ
sentence_scores = defaultdict(int)
for docid, num, wdcount, sentence in sentences:
    words = word_tokenize(sentence.lower())
    for word in words:
        if word in word_freq:
            sentence_scores[(docid, num, wdcount, sentence)] += word_freq[word]

# Chọn các câu có điểm cao nhất để tóm tắt
summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:20]

# Sắp xếp lại các câu theo thứ tự xuất hiện trong văn bản gốc
summary_sentences = sorted(summary_sentences, key=lambda x: sentences.index(x))

# Ghi nội dung tóm tắt vào file đầu ra
with open('d061joutput.txt', 'w') as file:
    for docid, num, wdcount, sentence in summary_sentences:
        file.write(f'<s docid="{docid}" num="{num}" wdcount="{wdcount}"> {sentence}</s>\n')