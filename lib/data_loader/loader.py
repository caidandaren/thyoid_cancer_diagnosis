


class Loader:
    def __init__(self, dhw=xxx, val_select=3, low_memory=True, crop = 32):
        assert val_select in [0,1,2,3]
        self.dhw = dhw
        self.crop = crop
        self.dataset = Dataset(low_memory=low_memory, iterable=True)


    @classmethod
    def get_balanced_loader(cls, batch_sizes, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        total_size = len(dataset)
        print('Size', total_size)
        index_generators = []
        for l_idx in range(len(batch_sizes)):
            # this must be list, or `l_idx` will not be eval
            iterator = [i for i in range(total_size) if dataset.label[i, l_idx]]
            index_generators.append(shuffle_iterator(iterator))
        while True:
            data = []
            for i, batch_size in enumerate(batch_sizes):
                generator = index_generators[i]
                for _ in range(batch_size):
                    idx = next(generator)
                    data.append(dataset[idx])
            yield dataset._collate_fn(data)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)
