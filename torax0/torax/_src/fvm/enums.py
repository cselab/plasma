import enum


@enum.unique
class InitialGuessMode(enum.StrEnum):
    X_OLD = 'x_old'
    LINEAR = 'linear'
