from utils import *
import mindspore


class MLUserLoading(mindspore.nn.Cell): # sharing
    def __init__(self, embedding_size, name):
        super(MLUserLoading, self).__init__(auto_prefix=True)
        self.num_gender = config['n_gender']
        self.num_age = config['n_age']
        self.num_occupation = config['n_occupation']
        self.embedding_size = embedding_size

        self.embedding_gender = mindspore.nn.Embedding(vocab_size=self.num_gender, embedding_size=self.embedding_size)
        self.embedding_age = mindspore.nn.Embedding(vocab_size=self.num_age,embedding_size=self.embedding_size)
        self.embedding_occupation = mindspore.nn.Embedding(vocab_size=self.num_occupation,embedding_size=self.embedding_size)
        self.name = name

    def construct(self, x1):
        gender_idx, age_idx, occupation_idx = x1[:,0], x1[:,1], x1[:,2]
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        concat_emb = mindspore.ops.concat((gender_emb, age_emb, occupation_emb), 1)
        return concat_emb


class MLItemLoading(mindspore.nn.Cell): # sharing
    def __init__(self, embedding_size, name):
        super(MLItemLoading, self).__init__(auto_prefix=True)
        self.rate_dim = config['n_rate']
        self.genre_dim = config['n_genre']
        self.director_dim = config['n_director']
        self.year_dim = config['n_year']
        self.embedding_size = embedding_size
        self.name = name

        self.emb_rate = mindspore.nn.Embedding(vocab_size=self.rate_dim, embedding_size=self.embedding_size)
        self.emb_genre = mindspore.nn.Dense(in_channels=self.genre_dim, out_channels=self.embedding_size)
        self.emb_director = mindspore.nn.Dense(in_channels=self.director_dim, out_channels=self.embedding_size)
        self.emb_year = mindspore.nn.Embedding(vocab_size=self.year_dim, embedding_size=self.embedding_size)

    def construct(self, x2):
        rate_idx, year_idx, genre_idx, director_idx = x2[:,0], x2[:,1], x2[:,2:27], x2[:,27:]
        rate_emb = self.emb_rate(rate_idx)
        year_emb = self.emb_year(year_idx)
        genre_emb = mindspore.ops.Sigmoid()(self.emb_genre(genre_idx.float()))
        director_emb = mindspore.ops.Sigmoid()(self.emb_director(director_idx.float()))
        concat_emb = mindspore.ops.Concat(axis=1)((rate_emb, year_emb, genre_emb, director_emb))
        return concat_emb

class BKUserLoading(mindspore.nn.Cell):
    def __init__(self, embedding_size, name):
        super(BKUserLoading, self).__init__(auto_prefix=True)
        self.age_dim = config['n_age_bk']
        self.location_dim = config['n_location']
        self.embedding_size = embedding_size

        self.emb_age = mindspore.nn.Embedding(vocab_size=self.age_dim, embedding_size=self.embedding_size)
        self.emb_location = mindspore.nn.Embedding(vocab_size=self.location_dim, embedding_size=self.embedding_size)
        self.name = name

    def construct(self, x1):
        age_idx, location_idx = x1[:,0], x1[:,1]
        age_emb = self.emb_age(age_idx)
        location_emb = self.emb_location(location_idx)
        concat_emb = mindspore.ops.operations.Concat((age_emb, location_emb), 1)
        return concat_emb

class BKItemLoading(mindspore.nn.Cell):
    def __init__(self, embedding_size, name):
        super(BKItemLoading, self).__init__(auto_prefix=True)
        self.year_dim = config['n_year_bk']
        self.author_dim = config['n_author']
        self.publisher_dim = config['n_publisher']
        self.embedding_size = embedding_size

        self.emb_year = mindspore.nn.Embedding(vocab_size=self.year_dim, embedding_size=self.embedding_size)
        self.emb_author = mindspore.nn.Embedding(vocab_size=self.author_dim, embedding_size=self.embedding_size)
        self.emb_publisher = mindspore.nn.Embedding(vocab_size=self.publisher_dim, embedding_size=self.embedding_size)
        self.name = name

    def construct(self, x2):
        author_idx, year_idx, publisher_idx = x2[:,0], x2[:,1], x2[:,2]
        year_emb = self.emb_year(year_idx)
        author_emb = self.emb_author(author_idx)
        publisher_emb = self.emb_publisher(publisher_idx)
        concat_emb = mindspore.ops.operations.Concat((year_emb, author_emb, publisher_emb), 1)
        return concat_emb

