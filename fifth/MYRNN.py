import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
import spacy


def IMDB_dataset(device):
    # 准备数据
    # 首先，我们要创建两个Field 对象：这两个对象包含了我们打算如何预处理文本数据的信息。
    # spaCy:英语分词器,类似于NLTK库，如果没有传递tokenize参数，则默认只是在空格上拆分字符串。
    # torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
    TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language='en_core_web_sm')
    LABEL = data.LabelField(dtype=torch.float)
    # 划分训练集和测试集
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # 将数据集划分为训练集和验证集
    train_data, valid_data = train_data.split(split_ratio=0.8)
    """
    创造词典
    从预训练的词向量（vectors）中，将当前(corpus语料库)词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）
    预训练的 vectors 来自glove模型，每个单词有100维。glove模型训练的词向量参数来自很大的语料库
    而电影评论的语料库小一点，所以词向量需要更新，glove的词向量适合用做初始化参数。    
    """
    maxVocabSize = 25000
    TEXT.build_vocab(train_data, max_size=maxVocabSize, vectors='glove.6B.100d')
    LABEL.build_vocab(train_data)
    batchSize = 32
    # 创造数据迭代器

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batchSize,
        sort_within_batch=True,
        device=device
    )
    return train_iterator, valid_iterator, test_iterator, len(TEXT.vocab), TEXT


class MyRNN(nn.Module):
    def __init__(self, input, embedding, hidden, output):
        super().__init__()
        # embedding的作用就是将每个单词变成一个词向量
        # vocab_size=input词汇表长度，embedding每个单词的维度
        self.embedding = nn.Embedding(input, embedding)
        self.rnn = nn.RNN(embedding, hidden)
        self.fc = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        predictions = self.relu(self.fc(hidden.squeeze(0)))
        predictions = self.fc2(predictions)
        return predictions



def binary_accuracy(preds, y):
    round_preds = torch.round((torch.sigmoid(preds)))
    corrrect = (round_preds == y).float()
    accuracy = corrrect.sum() / len(corrrect)
    return accuracy


def train_epoch(model, trainIter, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for batch in trainIter:
        optimizer.zero_grad()
        predictions = model(batch.text[0]).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()
    return epoch_loss / len(trainIter), epoch_accuracy / len(trainIter)


def evaluate(model, Iter, criterion):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    with torch.no_grad():
        for batch in Iter:
            predictions = model(batch.text[0]).squeeze(1)
            loss = criterion(predictions, batch.label)
            accuracy = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
    return epoch_loss / len(Iter), epoch_accuracy / len(Iter)


def train(model, trainIter, validIter, optimizer, criterion, epochs):
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(model, trainIter, optimizer, criterion)
        validation_loss, validation_accuracy = evaluate(model, validIter, criterion)
        print('Epoch:[{}/{}]'.format(epoch + 1, epochs))
        print('Train Loss : {:.3f}  | Train Accuracy : {:.3f}%'.format(train_loss, train_accuracy * 100))
        print('Validation Loss : {:.3f}  | Validation Accuracy : {:.3f}%'.format(validation_loss,
                                                                                 validation_accuracy * 100))


def predict(model, sentence, TEXT, device, threshold=0.5):
    nlp = spacy.load('en_core_web_sm')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = torch.unsqueeze(tensor, 1)
    prediction = torch.sigmoid((model(tensor)))
    if prediction.item() > threshold:
        return 'Positive'
    else:
        return 'Negative'


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('----------------loading--------------')
    train_iterator, valid_iterator, test_iterator, input, Text = IMDB_dataset(device)
    embedding = 100
    hiddden = 256
    output = 1
    model = MyRNN(input, embedding, hiddden, output).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    epochs = 10
    # ------------训练-----------------
    print('----------------training--------------')
    train(model, train_iterator, valid_iterator, optimizer, criterion, epochs)
    # --------------测试-----------------
    print('-----------------Testing--------------------')
    test_loss, test_accuracy = evaluate(model, test_iterator, criterion)
    print(
        'Test Loss : {:.3f}  | Test Accuracy : {:.3f}%'.format(test_loss, test_accuracy * 100))
    # ----------------预测---------------
    print('-------------------predicting-------------------')
    positive_review = 'I love this movie! It is awesome '
    negative_review = 'This movie is terrible. I hated it'
    print('Positive review prediction: {}'.format(predict(model, positive_review, Text, device)))
    print('Negative review prediction: {}'.format(predict(model, negative_review, Text, device)))
