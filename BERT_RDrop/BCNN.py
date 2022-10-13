"""
IDCNN(空洞CNN) 当卷积Conv1D的参数dilation_rate>1的时候，便是空洞CNN的操作
"""
import keras.layers
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Embedding, Dense, Dropout, Input, LSTM
from keras.layers import Conv1D
from keras_contrib.layers import CRF
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# set_session(tf.Session(config=config))
class IDCNNCRF(object):
    def __init__(self,
                 vocab_size: int,  # 词的数量(词表的大小)
                 n_class: int,  # 分类的类别(本demo中包括小类别定义了7个类别)
                 max_len: int = 100,  # 最长的句子最长长度
                 embedding_dim: int = 128,  # 词向量编码长度
                 drop_rate: float = 0.5,  # dropout比例
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate
        pass

    def creat_model(self):
        """
        本网络的机构采用的是，
           Embedding
           直接进行2个常规一维卷积操作
           接上一个空洞卷积操作
           连接全连接层
           最后连接CRF层

        kernel_size 采用2、3、4

        cnn  特征层数: 64、128、128
        """

        inputs = Input(shape=(self.max_len,))

        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)

        x10 = x(inputs)#原始


        xx1 = keras.layers.Lambda(lambda x: K.reverse(x, axes=2))(x10)



        x1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=1)(x10)
        xx1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=1)(xx1)



        x1 = Dropout(0.5)(x1)
        xx1 = Dropout(0.5)(xx1)


        '''----------'''

        x1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='causal',
                    dilation_rate=1)(x1)
        xx1 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='causal',
                     dilation_rate=2)(xx1)


        x1 = Dropout(0.5)(x1)
        xx1 = Dropout(0.5)(xx1)

        '''--------------'''

        x1 = Conv1D(filters=128,
                    kernel_size=3,
                    activation='relu',
                    padding='same',
                    dilation_rate=4)(x1)
        xx1 = Conv1D(filters=128,
                     kernel_size=3,
                     activation='relu',
                     padding='same',
                     dilation_rate=4)(xx1)



        xx1 = keras.layers.Lambda(lambda x: K.reverse(x, axes=2))(xx1)



        x1 = keras.layers.Add()([x1, xx1])


        x1 = keras.layers.Concatenate()([x1, x10])
        # x1 = keras.layers.Concatenate()([x1, x2])




        x1 = Dropout(self.drop_rate)(x1)

        x1 = Dense(self.n_class)(x1)

        self.crf = CRF(self.n_class, sparse_target=False)
        x1 = self.crf(x1)
        self.crf1 = CRF(5, sparse_target=False)

        self.model = Model(inputs=inputs, outputs=x1)
        self.model.summary()
        self.compile()
        return self.model

    def compile(self):
        self.model.compile('adam',
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])


if __name__ == '__main__':
    import sys
    from datetime import datetime as dt
    from args import read_options
    from bert4keras.backend import keras, K
    from bert4keras.tokenizers import Tokenizer
    from bert4keras.models import build_transformer_model
    from bert4keras.optimizers import Adam
    from bert4keras.snippets import sequence_padding, DataGenerator
    from bert4keras.snippets import open
    from keras.layers import Lambda, Dense, Bidirectional, CuDNNLSTM, LSTM
    from keras.utils import to_categorical
    from tqdm import tqdm
    from keras.losses import kullback_leibler_divergence as kld
    import warnings
    from keras_contrib.layers import CRF
    import random
    random.seed(111)
    import numpy as np
    np.random.seed(111)


    warnings.filterwarnings("ignore")

    sys.path.append('../')
    ts = dt.now()
    args = read_options()

    from DataProcess.process_data import DataProcess
    from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score
    import numpy as np
    from keras.utils.vis_utils import plot_model

    config_path = '../chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_config.json'
    # checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
    checkpoint_path = '../chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_model.ckpt'
    # dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'
    dict_path = '../chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/vocab.txt'
    dataname = 'msra'
    # train_frac = 0.01 #数据比例
    #NER 数据集
    dp = DataProcess(max_len=100, data_type=dataname)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)
    train_datas = train_data
    train_datas[0] = train_data[0][0:10000]
    train_datas[1] = train_data[1][:10000]
    train_label = train_label[:5000]

    # 文本分类数据集
    def load_data(filename):
        """加载数据
        单条格式：(文本, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                Ls = l.strip().split(',')
                text, label = Ls[2], Ls[1]
                D.append((text, int(label)))
        return D

    anly_data = load_data('../anayle_data/anayle_train.txt')


    # anly_data = [(t, 0) for t, l in anly_data[:]]
    # anly_train = dp.anly_bert(anly_data)
    # anly_label = dp.anly_label_to_one_hot(anly_data)

    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    class data_generator_rdrop(DataGenerator):
        """数据生成器
        """

        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []
            for is_end, (text, label) in self.sample(random):
                token_ids, segment_ids = tokenizer.encode(text, maxlen=100)
                for i in range(2):
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    batch_labels.append(label)
                if len(batch_token_ids) == self.batch_size * 2 or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = to_categorical(batch_labels, 2)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    # 输出的 是 padding 之后的 句子表示，这个表示是分词器给的， 然后是文本表示，都是一个句子，就都是0， 还有onehot的标签[1,0]，因为是二分类
                    # 但是 32的batch 重复了一遍， 是64维度的 重复的方式是[1,1,2,2,3,3....]
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    # NER model
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        dropout_rate=0.3,
        return_keras_model=False,
    )
    # NER model
    # outs = Dense(768,kernel_initializer=bert.initializer)(bert.model.output)
    ner_output = Bidirectional(LSTM(128, return_sequences=True))(bert.model.output)
    ner_output = Dense(7)(ner_output)
    crf = CRF(7, sparse_target=False)
    ner_output = crf(ner_output)

    # r-drop model
    r_output = Lambda(lambda x: x[:, 0])(bert.model.output)  # 在这里取cls
    # output = Dense(
    #     units=2,
    #     activation='softmax',
    #     kernel_initializer=bert.initializer
    # )(outs)
    ner_model = keras.models.Model(bert.model.input, ner_output)
    ner_model.summary()
    ner_model.compile(optimizer=Adam(1e-5),
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])


    def kld_rdrop(y_true, y_pred):
        """无监督部分只需训练KL散度项
        """
        loss = (kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2]))
        return K.mean(loss)*4


    model_rdrop = keras.models.Model(bert.model.input, r_output)
    model_rdrop.compile(
        loss=kld_rdrop,
        optimizer=Adam(1e-5),
    )


    class Evaluator(keras.callbacks.Callback):
        """评估与保存
        """

        def __init__(self):
            self.best_val_acc = 0.
            self.x = anly_data
            self.falg = False
            # self.y = anly_label
            self.data = data_generator_rdrop(anly_data, 8).forfit()
            # self.data = data_generator_rdrop(unlabeled_data, batch_size).forfit()

        def on_epoch_end(self, epoch, logs=None):
            # val_acc = evaluate(valid_generator)
            # if val_acc > self.best_val_acc:
            #     self.best_val_acc = val_acc
            #     model.save_weights('best_model.weights')
            # test_acc = evaluate(test_generator)
            # model_rdrop.fit(self.x,self.y)
            # print(epoch)
            pass
        def on_batch_end(self, batch, logs=None):
            # print(batch)
            flag = True
            # if flag:
            #     if batch % 50 == 0 and batch != 0:
            #         model_rdrop.fit(self.x, self.y)
            # else:
            #     pass

            # if flag:
                # if batch % 50 == 0 and batch != 0:
            #     if batch == 1 or batch == 2:
            #         dx, dy = next(self.data)
            #         model_rdrop.train_on_batch(dx, dy)
            # else:
            #     print('this batch no rdrop')

            # pass
            #
            # dx, dy = next(self.data)
            # model_rdrop.train_on_batch(dx, dy)
            if self.falg:
                dx, dy = next(self.data)
                model_rdrop.train_on_batch(dx, dy)
            else:
                pass

            # pass




            # model_rdrop.train_on_batch(self.x,self.y)
    evaluator = Evaluator()
    train_data[0] = train_data[0][:1000]
    train_data[1] = train_data[1][:1000]


    ner_model.fit(
        train_data,train_label[:1000],batch_size=8,

        epochs=20,
        callbacks=[evaluator]
    )


    # lstm_crf = IDCNNCRF(vocab_size=dp.vocab_size, n_class=7, max_len=100)
    # lstm_crf.creat_model()
    # model = lstm_crf.model
    # train_data = train_data[0:5000]
    # train_label = train_label[0:5000]
    #
    # lens = len(train_data)
    # print(lens)
    #
    # model.fit(train_data, train_label, batch_size=64, epochs=15,
    #           validation_data=[test_data, test_label])

    # 对比测试数据的tag
    y = ner_model.predict(test_data)
    #
    label_indexs = []
    pridict_indexs = []

    num2tag = dp.num2tag()
    i2w = dp.i2w()
    texts = []
    texts.append(f"字符\t预测tag\t原tag\n")
    for i, x_line in enumerate(test_data[0]):
        for j, index in enumerate(x_line):
            if index != 0:
                char = i2w.get(index, ' ')
                t_line = y[i]
                t_index = np.argmax(t_line[j])
                tag = num2tag.get(t_index, 'O')
                pridict_indexs.append(t_index)

                t_line = test_label[i]
                t_index = np.argmax(t_line[j])
                org_tag = num2tag.get(t_index, 'O')
                label_indexs.append(t_index)

                texts.append(f"{char}\t{tag}\t{org_tag}\n")
        texts.append('\n')
    names = r'rdopNERfalse' + '.txt'

    log = open(names, 'w', encoding='utf-8')
    for i in texts:
        log.write(i)
    log.close()
    f1score = f1_score(label_indexs, pridict_indexs, average='macro')
    recalls = recall_score(label_indexs, pridict_indexs, average='macro')
    precision = precision_score(label_indexs, pridict_indexs, average='macro')
    accuracy = accuracy_score(label_indexs, pridict_indexs)
    accuracy = str(accuracy)
    print(f"f1score:{f1score}")
    print(f"recall:{recalls}")
    print(f"pre :{precision}")
    print(f"acc :{accuracy}")
    te = dt.now()
    spent = te - ts
    print('Time spend : %s' % spent)
    #
    with open('all_logs.txt','a',encoding='utf-8') as f:
        f.write(str(evaluator.falg))
        f.write('\n')
        f.write(str(len(train_data[0])))
        f.write('\n')
        f.write('f1: '+ str(f1score))
        f.write('\n')
        f.write('pre: '+ str(precision))
        f.write('\n')
        f.write('recall: '+str(recalls))
        f.write('\n')
        f.write('acc: '+str(accuracy))
        f.write('\n')


    '''

500 epo 20
T
f1score:0.8117341633882698
recall:0.8094621448137735
pre :0.8221553533185565
acc :0.9869740105683166

f1score:0.7994080619730405
recall:0.8088340487649716
pre :0.798034865457342
acc :0.9849800496063841

f1score:0.8047393483105242
recall:0.802416463017501
pre :0.8177846261746148
acc :0.9863905963550091

f1score:0.8013866718599078
recall:0.7884923594185247
pre :0.8265019395160295
acc :0.9864283403429311

f1score:0.8038271043892938
recall:0.7915449124783314
pre :0.8243456209694823
acc :0.9867497034400949

f1score:0.7906043139649453
recall:0.7877150314148939
pre :0.8103677312158661
acc :0.9861511916316187

f1score:0.7993244234999113
recall:0.7974944277932758
pre :0.8140115648891791
acc :0.9861156044430066

f1score:0.7858584950071295
recall:0.7830808499156212
pre :0.8059780276703629
acc :0.9855968942089939

f1score:0.792920381984417
recall:0.787691031566169
pre :0.811272003856591
acc :0.9857618893561954


f1score:0.8089421706294805
recall:0.805242382587213
pre :0.8219518762812142
acc :0.9864035371508681

f1score:0.787396329331098
recall:0.7892271625757858
pre :0.8009359771502175
acc :0.9853294510945757

f1score:0.8000765514614185
recall:0.8059398244128417
pre :0.8017200633216914
acc :0.9856400301951903



F
f1score:0.7973458779785291
recall:0.804104071922107
pre :0.8011430686690735
acc :0.9850501455839534

f1score:0.795256223697408
recall:0.8036729874938128
pre :0.7999866573272632
acc :0.9854534670548906

f1score:0.7797502169431058
recall:0.7951202727646575
pre :0.7869406958258544
acc :0.9836471476329127

f1score:0.7855119998475723
recall:0.7987496689928355
pre :0.7878888184002844
acc :0.9847773104712606

f1score:0.778385610000002
recall:0.8019279172202717
pre :0.7689995157936107
acc :0.9834217621050362

f1score:0.7783377101205698
recall:0.7972979523939517
pre :0.7730042262457731
acc :0.9825752183759301

f1score:0.7850701192861888
recall:0.7979176107971997
pre :0.7901561280226405
acc :0.9848970128329559


f1score:0.7784043032225051
recall:0.799518760681357
pre :0.7719014494592601
acc :0.9835037204788094

f1score:0.7893087059036167
recall:0.8066566395497504
pre :0.7866242687584653
acc :0.9845443761457996

f1score:0.7910560174443303
recall:0.8075172816106496
pre :0.7879581913729551
acc :0.9834282325029656


'''











