#! -*- coding:utf-8 -*-
# 通过R-Drop进行半监督学习
# 在苏神的R_DROP上的NER应用
# 苏神博客：https://kexue.fm/archives/7466

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense,Bidirectional,CuDNNLSTM,LSTM
from keras.utils import to_categorical

from keras.losses import kullback_leibler_divergence as kld
import warnings
from keras_contrib.layers import CRF
warnings.filterwarnings("ignore")

# 配置信息
num_classes = 2
maxlen = 128
batch_size = 8
train_frac = 0.01  # 标注数据的比例
use_rdrop = True  # 可以比较True/False的效果

# BERT base
# config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    """加载数据
    MSRA NER 的数据格式
    """
    labels = {'O':0,'B-PER':1,'I-PER':2,'B-LOC':3,'I-LOC':4,'B-ORG':5,'I-ORG':6}
    text = []
    label = []
    t = ''
    l = []
    with open(filename, encoding='utf-8') as f:
        ff = f.readlines()
        for i in ff:
            if i == '\n':
                text.append((t,l))
                t = ''
                l = []
                continue
            i = i.strip()
            t += i.split(' ')[0]
            l.append(labels.get(i.split(' ')[1]))


    return text


# 加载数据集
train_data = load_data('../NER_data/train.txt')

test_data = load_data('../NER_data/test.txt')



# 模拟标注和非标注数据
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 0) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels,batch_labelss = [], [], [],[]
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labelss.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                for i in batch_labelss:
                    i = to_categorical(i, 7)
                    i = i.tolist()

                    # i = sequence_padding(i)
                    if isinstance(i,list) and isinstance(batch_labels,list):
                        batch_labels.append(i)
                    else:
                        batch_labels = batch_labels.tolist()
                        batch_labels.append(i)

                # batch_labels = to_categorical(batch_labels, num_classes)
                batch_labels = sequence_padding(batch_labels,batch_token_ids.shape[1])
                # print(batch_token_ids.shape, batch_labels.shape)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labelss,batch_labels = [], [], [],[]


class data_generator_rdrop(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = to_categorical(batch_labels, num_classes)
                batch_labels = sequence_padding(batch_labels,batch_token_ids.shape[1])
                # print(batch_token_ids.shape,batch_labels.shape)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                # 输出的 是 padding 之后的 句子表示，这个表示是分词器给的， 然后是文本表示，都是一个句子，就都是0， 还有onehot的标签[1,0]，因为是二分类
                # 但是 32的batch 重复了一遍， 是64维度的 重复的方式是[1,1,2,2,3,3....]
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
# valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
    return_keras_model=False,
)
#NER model
# ner_output = Bidirectional(LSTM(128,return_sequences=True))(bert.model.output)
ner_output = Dense(7)(bert.model.output)
# ner_output = Dense(7)(ner_output)
crf = CRF(7, sparse_target=False)
ner_output = crf(ner_output)


# r-drop model
r_output = Lambda(lambda x: x[:, 0])(bert.model.output) #在这里取cls
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(r_output)

# 用于正常训练的模型
model = keras.models.Model(bert.model.input, ner_output)
model.summary()

model.compile(optimizer=Adam(1e-5),
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])


def kld_rdrop(y_true, y_pred):
    """无监督部分只需训练KL散度项
    """
    loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return K.mean(loss)


# 用于R-Drop训练的模型
model_rdrop = keras.models.Model(bert.model.input, output)
model_rdrop.compile(
    loss=kld_rdrop,
    optimizer=Adam(1e-5),
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        # y_true = y_true.argmax(axis=1)
        # y_true = y_true.tolist()
        # print(y_true.shape)
        # print(y_pred.shape)

        y_t = y_true.reshape(y_true.shape[1] * y_true.shape[0],7)

        y_p = y_pred.reshape(y_true.shape[1] * y_true.shape[0], 7)

        y_t = y_t.tolist()
        y_p = y_p.tolist()

        total += len(y_p)
        for i in range(int(len(y_p))):
            if y_t[i] == y_p[i]:
                right += 1

        # for i in
    print('right:',right)
    print('total:',total)


    return  right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.
        self.data = data_generator_rdrop(unlabeled_data, batch_size).forfit()

    def on_epoch_end(self, epoch, logs=None):
        # val_acc = evaluate(valid_generator)
        # if val_acc > self.best_val_acc:
        #     self.best_val_acc = val_acc
        #     model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u' test_acc: %.5f\n' %
            ( test_acc)
        )

    def on_batch_end(self, batch, logs=None):
        if use_rdrop:
            dx, dy = next(self.data)
            model_rdrop.train_on_batch(dx, dy)


if __name__ == '__main__':

    evaluator = Evaluator()
    x =  train_generator.forfit()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=30,
        epochs=15,
        callbacks=[evaluator]
    )



else:

    model.load_weights('best_model.weights')
