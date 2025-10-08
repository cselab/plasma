import abc
class BaseQLKNNModel(abc.ABC):

    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
