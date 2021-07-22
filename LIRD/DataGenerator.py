import csv
import random
import pandas as pd


class DataGenerator:
    def __init__(self, data_path, item_path):
        """
        Load data from the DB MovieLens
        List the users and the items
        List all the users historic
        """
        self.data = self.load_data(data_path, item_path)
        self.users = self.data['userId'].unique()  # list of all users
        self.items = self.data['itemId'].unique()  # list of all items
        self.user_rate_history = self.generate_history()
        # 用于训练/测试的userId列表
        self.user_for_train = None
        self.user_for_test = None
        self.train = []
        self.test = []

    @staticmethod
    def load_data(data_path, item_path):
        """
        载入数据并根据itemId进行左连接, 最终返回(userId, itemId, rating, timestamp, itemName)
        Load the data and merge the name of each movie. A row corresponds to a rate given by a user to a movie.

        :param data_path: string
                            path to the data 100k MovieLens
                            contains usersId;itemId;rating
        :param item_path: string
                            path to the data 100k MovieLens
                            contains itemId;itemName
        :return: DataFrame (userId, itemId, rating, timestamp, itemName)
                Contains all the ratings
        """
        data = pd.read_csv(data_path, sep='\t', names=['userId', 'itemId', 'rating', 'timestamp'])
        movie_titles = pd.read_csv(item_path, sep='|', names=['itemId', 'itemName'], usecols=range(2),
                                   encoding='latin-1')
        return data.merge(movie_titles, on='itemId', how='left')

    def generate_history(self):
        """
        Group all rates given by users and store them from older to most recent.

        :return: List(DataFrame) 每个用户的历史评分信息 [(userId, itemId, rating, timestamp, itemName)]
        """
        historic_users = []
        for index, user_id in enumerate(self.users):
            # 筛出该user的评分记录
            temp = self.data[self.data['userId'] == user_id]
            # 根据时间戳升序排序并重新计算索引
            temp = temp.sort_values('timestamp').reset_index()
            # 原地抛掉原始索引列
            temp.drop('index', axis=1, inplace=True)
            historic_users.append(temp)
        return historic_users

    @staticmethod
    def sample_history(user_session, state_ratio=0.8, max_samp_by_user=5, max_state=100, max_action=50,
                       state_nums=[], action_nums=[]):
        """
        从每个session中进行取样生成一个或多个样本
        如果nb_states、nb_actions为空,则从session中随机取1-max_samp_by_user个样本,且每个样本的state,action长度不等
        否则,
            从一个session的前state_ratio部分,随机取样nb_state个(item,rating)作为一个样本的state,
            从其后(1-state_ratio)部分,随机取样nb_action个(item,rating)作为一个样本的action

        :param user_session:  用户session, [(userId, itemId, rating, timestamp, itemName), ... ]
        :param state_ratio:
        :param max_samp_by_user: 从一个session中,最多取max_samp_by_user个(state,action)样本
        :param max_state: 最多取max_state个(item,rating)作为一个样本的state
        :param max_action: 最多取max_action个(item,rating)作为一个样本的action
        :param state_nums: 从session的前state_ratio部分,随机取样state_num_i个(item,rating)作为一个样本的state
        :param action_nums: 从session的后(1-state_ratio)部分,随机取样action_num_i个(item,rating)作为一个样本的action
        :return:
        """
        n = len(user_session)
        sep = int(state_ratio * n)
        nb_sample = random.randint(1, max_samp_by_user)
        if not state_nums:
            state_nums = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
        if not action_nums:
            action_nums = [min(random.randint(1, n - sep), max_action) for i in range(nb_sample)]
        assert len(state_nums) == len(action_nums), 'Given array must have the same size'

        states = []
        actions = []
        # SELECT SAMPLES IN HISTORY
        for i in range(len(state_nums)):
            sample_states = user_session.iloc[0:sep].sample(state_nums[i])
            sample_actions = user_session.iloc[-(n - sep):].sample(action_nums[i])

            sample_state = []
            sample_action = []
            for j in range(state_nums[i]):
                row = sample_states.iloc[j]
                # FORMAT STATE
                state = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_state.append(state)

            for j in range(action_nums[i]):
                row = sample_actions.iloc[j]
                # FORMAT ACTION
                action = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_action.append(action)

            states.append(sample_state)
            actions.append(sample_action)
        return states, actions

    def generate_train_test(self, train_ratio, seed=None):
        """
        Shuffle the historic of users and separate it in a train and a test set.
        Store the ids for each set.
        An user can't be in both set.

        :param train_ratio: 用于训练的样本比例
        :param seed: 随机数种子,用于打乱用户数据
        :return:
        """
        n = len(self.user_rate_history)

        if seed is not None:
            random.Random(seed).shuffle(self.user_rate_history)
        else:
            random.shuffle(self.user_rate_history)

        self.train = self.user_rate_history[:int((train_ratio * n))]
        self.test = self.user_rate_history[int((train_ratio * n)):]
        # iloc为取指定位置元素, 此处即取出train/test数据集中的userId分别放入user_for_train, user_for_test
        self.user_for_train = [h.iloc[0, 0] for h in self.train]
        self.user_for_test = [h.iloc[0, 0] for h in self.test]

    def write_csv(self, file_name, users_session, delimiter=';', state_ratio=0.8, max_samp_by_user=5, max_state=100,
                  max_action=50, states_nums=[], actions_nums=[]):
        """
        基于给定的多个session,根据states_nums,actions_nums从session中取样
        生成的文件内容为    state;action_reward;n_state
        例如: states_nums=2, actions_nums=1
        则: itemId_1&rating_1|itemId_2&rating_2; itemId_3&rating_3; itemId_1&rating_1|itemId_2&rating_2|itemId_3&rating_3

        state_ratio=0.8, 则从每个session的前0.8取states_num个样例作为state, 从session的后0.2取actions_num个样例作为action

        :param file_name: string, path to the file to be produced
        :param users_session: [session1, session2, ...]
        :param delimiter: string, delimiter for the csv
        :param state_ratio: float, 取样action的比例
        :param max_samp_by_user: 从一个session中,最多取max_samp_by_user个(state,action)样本
        :param max_state: 最多取max_state个(item,rating)作为一个样本的state
        :param max_action: 最多取max_action个(item,rating)作为一个样本的action
        :param states_nums: 从session的前state_ratio部分,随机取样state_num_i个(item,rating)作为一个样本的state
        :param actions_nums: 从session的后(1-state_ratio)部分,随机取样action_num_i个(item,rating)作为一个样本的action
        :return:
        """
        with open(file_name, mode='w') as file:
            f_writer = csv.writer(file, delimiter=delimiter)
            f_writer.writerow(['state', 'action_reward', 'n_state'])
            for user_histo in users_session:
                states, actions = self.sample_history(user_histo, state_ratio, max_samp_by_user, max_state, max_action,
                                                      states_nums, actions_nums)
                for i in range(len(states)):
                    # FORMAT STATE
                    state_str = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str = '|'.join(actions[i])
                    # FORMAT N_STATE
                    n_state_str = state_str + '|' + action_str
                    f_writer.writerow([state_str, action_str, n_state_str])
