import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

#构建词表
def build_vocab():
    # 数据集位置
    file_name = 'data/jaychou_lyrics.txt'
    # 分词结果存储位置
    unique_words = []
    all_words = []
    # 遍历数据集中的每一行文本
    for line in open(file_name, 'r', encoding='utf-8'):  # 新增encoding=utf-8避免中文乱码
        # 使用jieba分词, 分割结果是一个列表
        words = jieba.lcut(line)
        # 所有的分词结果存储到all_sentences，其中包含重复的词组
        all_words.append(words)
        # 遍历分词结果，去重后存储到unique_words
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    # 语料中词的数量
    word_count = len(unique_words)

    # 词到索引映射
    #使用 enumerate 遍历 unique_words，给每一个单词word标注一个下标idx
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}#字典推导式
    # 词表索引表示
    corpus_idx = []
    # 遍历每一行的分词结果
    for words in all_words:
        temp = []
        # 获取每一行的词，并获取相应的索引
        for word in words:
            temp.append(word_to_index[word])
        # 在每行词之间添加空格隔开
        temp.append(word_to_index[' '])
        # 获取当前文档中每个词对应的索引
        corpus_idx.extend(temp)
    return unique_words, word_to_index, word_count, corpus_idx

if __name__ == "__main__":
    # 获取数据
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    print("词的数量：\n", word_count)
    print("去重后的词:\n", unique_words)
    print("每个词的索引：\n", word_to_index)
    print("当前文档中每个词对应的索引：\n", corpus_idx)
#--------------------------------------------------

#构建数据集
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中词的索引
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数
        self.num_chars = num_chars
        # 词的数量
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count // self.num_chars

    def __len__(self):
        # 返回句子数量
        return self.number

#当你用dataset[0]/dataset[1]取数据时，这个方法会根据传入的idx（样本索引），返回一组训练样本：输入 x + 目标 y。
#而歌词生成的核心逻辑是：用前 N 个词，预测后 N 个词（每个位置的词预测下一个词） —— 这就是x和y的本质！
    def __getitem__(self, idx):
        # idx指词的索引，并将其修正索引值到文档的范围里面
        #max(数据1，数据2)，比较两者的较大值
        start = min(max(idx, 0), self.word_count - self.num_chars - 2)
        # 输入值
        x = self.corpus_idx[start:start + self.num_chars]
        # 网络预测结果（目标值）
        y = self.corpus_idx[start + 1:start + 1 + self.num_chars]
        # 返回结果
        return torch.tensor(x), torch.tensor(y)

    # 测试自定义Dataset
    dataset = LyricsDataset(corpus_idx, 5)
    x, y = dataset.__getitem__(0)
    print("\n网络输入值：", x)
    print("目标值：", y)

    # 测试DataLoader（批量加载）
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_x, batch_y in dataloader:
        print("\n批量输入形状：", batch_x.shape)
        print("批量目标形状：", batch_y.shape)
        break  # 只打印第一个批次


# (1)构建词表
# 分词jieba.lchut("文本")
# 去重得到unique_words
# 构建词表，字典推导式word_index = {word:idx for idx,word in enumerate(unique_words)}
# 预料转化成连续编号序列corpus_idx
# (2)构建lyricsDataset数据集
# init初始化，传入corpus_idx和num_chars
# 切成“输入x，目标y”的样本

# 模型构建
class TextGenerator(nn.Module):
    def __init__(self, word_count):
        super(TextGenerator, self).__init__()
        # 初始化词嵌入层: 词向量的维度为128
        self.ebd = nn.Embedding(word_count, 128)
        # 循环网络层: 词向量维度128, 隐藏向量维度128, 网络层数1
        self.rnn = nn.RNN(128, 128)
        # 输出层: 特征向量维度128与隐藏向量维度相同,词表中词的个数
        self.out = nn.Linear(128, word_count)

    def forward(self, inputs, hidden):
        # 输出维度: (batch, seq_len,词向量维度128)
        embed = self.ebd(inputs)
        # 修改维度: (seq_len, batch,词向量维度128)，没写 batch_first = 必须用 transpose (0,1) 交换维度
        #permute(a,b,c,d)交换多个维度
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)
        # 输入维度: (seq_len*batch,词向量维度) 输出维度: (seq_len*batch, 128)
        # 为什么这里是
        # output.shape[-1]，以前是
        # output.size(-1)？
        # 意思完全一样！只是写法不一样！
        # .shape[-1] → numpy
        # 风格，拿最后一维大小
        # .size(-1) → PyTorch
        # 风格，拿最后一维大小

        #RNN输出的是三维，而全连接层只接受二维
        output = self.out(output.reshape((-1,output.shape[-1])))
        # 网络输出结果
        return output, hidden

    def init_hidden(self, bs=1):
        # 隐藏层的初始化:[网络层数, batch, 隐藏层向量维度]
        return torch.zeros(1, bs, 128)

# 模型训练
def train():
    # 构建词典
    index_to_word, word_to_index, word_count, corpus_idx = build_vocab()
    # 数据集
    lyrics = LyricsDataset(corpus_idx, 32)
    # 初始化模型
    model = TextGenerator(word_count)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练轮数
    epoch = 10

    for epoch_idx in range(epoch):
        # 数据加载器
        lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=1)
        # 训练时间
        start = time.time()
        iter_num = 0 # 迭代次数
        # 训练损失
        total_loss = 0.0
        # 遍历数据集
        for x, y in lyrics_dataloader:
            # 隐藏状态的初始化
            hidden = model.init_hidden(bs=1)
            # 模型计算
            output, hidden = model(x, hidden)
            # 计算损失
            # y:[batch,seq_len]->[seq_len,batch]->[seq_len*batch]
            y = torch.transpose(y, 0, 1).contiguous().view(-1)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1 # 迭代次数加1
            total_loss += loss.item()
        # 打印训练信息
        print('epoch %3s loss: %.5f time %.2f' % (epoch_idx + 1, total_loss / iter_num, time.time() - start))
    # 模型存储
    torch.save(model.state_dict(), 'data/lyrics_model_%d.pth' % epoch)

# 预测函数
def predict(start_word, sentence_length):
    # 构建词典
    index_to_word, word_to_index, word_count, _ = build_vocab()
    # 构建模型
    model = TextGenerator(word_count)
    # 加载参数
    model.load_state_dict(torch.load('data/lyrics_model_10.pth'))
    # 隐藏状态
    hidden = model.init_hidden()
    # 将起始词转换为索引
    word_idx = word_to_index[start_word]
    # 产生的词的索引存放位置
    generate_sentence = [word_idx]
    # 遍历到句子长度，获取每一个词
    for _ in range(sentence_length):
        # 模型预测
        #[[word_idx]]模型只吃二维张量
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 获取预测结果
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)
    # 根据产生的索引获取对应的词，并进行打印
    for idx in generate_sentence:
        print(index_to_word[idx], end="")

if __name__ == "__main__":
    # 训练
    # train()
    # 预测
    predict('分手', 50)