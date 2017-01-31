#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    # load an input data
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                        digits.target)
    print('Train data: {}'.format(X_train.shape))
    print('Test data: {}'.format(X_test.shape))


if __name__ == '__main__':
    main()
