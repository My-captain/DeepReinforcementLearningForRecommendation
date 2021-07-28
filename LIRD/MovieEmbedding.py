import pandas as pd

import numpy as np
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout


class EmbeddingsGenerator:
    def __init__(self, train_users, data):
        self.train_users = train_users

        # preprocess
        self.data = data.sort_values(by=['timestamp'])
        # make them start at 0
        self.data['userId'] = self.data['userId'] - 1
        self.data['itemId'] = self.data['itemId'] - 1
        self.user_count = self.data['userId'].max() + 1
        self.movie_count = self.data['itemId'].max() + 1
        self.user_movies = {}  # list of rated movies by each user
        for userId in range(self.user_count):
            self.user_movies[userId] = self.data[self.data.userId == userId]['itemId'].tolist()
        self.m = self.model()

    def model(self, hidden_layer_size=100):
        m = Sequential()
        m.add(Dense(hidden_layer_size, input_shape=(1, self.movie_count)))
        m.add(Dropout(0.2))
        m.add(Dense(self.movie_count, activation='softmax'))
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m

    def generate_input(self, user_id):
        """
        每次从一个user的session中, 随机抽取一个movie及其上下文, 其中target为该movie的one-hot vector, 其上下文即长度相同的0-1向量
        :param user_id:
        :return: context, target 从一个user的session中,随机抽取的一个movie及其上下文
        """
        user_movies_count = len(self.user_movies[user_id])
        # 从session中随机选一个movie
        random_user_movie_index = np.random.randint(0, user_movies_count - 1)  # 避免选中session末尾的movie
        # 生成该movie的one-hot vector
        target = np.zeros((1, self.movie_count))
        target[0][self.user_movies[user_id][random_user_movie_index]] = 1
        # 生成该movie的上下文向量
        context = np.zeros((1, self.movie_count))
        context_movie_id_list = self.user_movies[user_id][:random_user_movie_index] + self.user_movies[user_id][random_user_movie_index + 1:]
        context[0][context_movie_id_list] = 1
        return context, target

    def train(self, epochs=300, batch_size=10000):
        """
        基于用户session训练embedding
        :param epochs:
        :param batch_size:
        :return:
        """
        print(f"Training Movies' Embedding...")
        for i in range(epochs):
            print(f'epoch: {i+1}/{epochs}')
            batch = [self.generate_input(user_id=np.random.choice(self.train_users) - 1) for _ in range(batch_size)]
            # 把一个batch中的所有
            x_train = np.array([b[0] for b in batch])
            y_train = np.array([b[1] for b in batch])
            self.m.fit(x_train, y_train, epochs=1, validation_split=0.5)

    def test(self, test_users, batch_size=100000):
        """
        在session测试集上测试movie embedding的准确率
        :param test_users: 用于测试的userId列表
        :param batch_size:
        :return: [loss, accuracy]
        """
        batch_test = [self.generate_input(user_id=np.random.choice(test_users) - 1) for _ in range(batch_size)]
        x_test = np.array([b[0] for b in batch_test])
        y_test = np.array([b[1] for b in batch_test])
        return self.m.evaluate(x_test, y_test)

    def save_embeddings(self, file_name):
        """
        Generates a csv file containing the vector embedding for each movie.
        """
        inp = self.m.input  # input placeholder
        outputs = [layer.output for layer in self.m.layers]  # all layer outputs
        functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function
        # functor = K.function([inp], outputs)  # evaluation function

        # append embeddings to vectors
        vectors = []
        for movie_id in range(self.movie_count):
            movie = np.zeros((1, 1, self.movie_count))
            movie[0][0][movie_id] = 1
            layer_outs = functor([movie])
            vector = [str(v) for v in layer_outs[0][0][0]]
            vector = '|'.join(vector)
            vectors.append([movie_id, vector])

        # saves as a csv file
        embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'item_id': 'int32'})
        embeddings.to_csv(file_name, sep=';', index=False)


class Embeddings:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def size(self):
        return self.item_embeddings.shape[1]

    def get_embedding_vector(self):
        return self.item_embeddings

    def get_embedding(self, item_index):
        return self.item_embeddings[item_index]

    def embed(self, item_list):
        return np.array([self.get_embedding(item-1) for item in item_list])


def read_file(data_path):
    """
    从csv中读取样本并整理数据格式
    :param data_path:
    :return: [state,n_state,action,reward]每个单元格为一个list
    """
    data = pd.read_csv(data_path, sep=';')
    for col in ['state', 'n_state', 'action_reward']:
        data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
    for col in ['state', 'n_state']:
        data[col] = [np.array([e[0] for e in l]) for l in data[col]]

    data['action'] = [[e[0] for e in l] for l in data['action_reward']]
    data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
    data.drop(columns=['action_reward'], inplace=True)

    return data


def read_embeddings(embeddings_path):
    """ Load embeddings (a vector for each item). """

    embeddings = pd.read_csv(embeddings_path, sep=';')

    return np.array([[np.float64(k) for k in e.split('|')]
                     for e in embeddings['vectors']])


