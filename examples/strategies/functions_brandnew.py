"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects
import talib
import talib as ta # 因为SingleFactorModelNonlinear.用到的是简写的ta，为了保持一致就再import一次talib as ta
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def percentileofscore(a, score):
    a = np.asarray(a)
    n = len(a)
    score = np.asarray(score)
    # Prepare broadcasting
    score = score[..., None]
    def count(x):
        return np.count_nonzero(x, -1)
    
    left = count(a < score)
    right = count(a <= score)
    plus1 = left < right
    perct = np.ma.filled((left + right + plus1) * (50. / n), np.nan)
    return perct


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return np.nan_to_num(1 / (1 + np.exp(-x1)))

def _tanh(x1):
    with np.errstate(over='ignore', under='ignore'):
        return np.nan_to_num(np.tanh(x1))

def _elu(x1):
    with np.errstate(over='ignore', under='ignore'):
        x = np.nan_to_num(np.where(x1 > 0, x1, 1 * (np.exp(x1) - 1)))
        return scaler(x)
def scaler_std(x, window=500):
    """
     std缩放
    :param x:
    :param window:
    :return:
    """
    x = x.astype(float)
    window1 = 200
    # 滚动缩尾
    x_mean = ta.MA(x, timeperiod=window1)
    sum_square = ta.MA(x ** 2, timeperiod=window1) * window1
    var = np.sqrt((sum_square - window1 * (x_mean ** 2)) / (window1 - 1))
    var[var <= 0] = 1e-8
    x_std = np.sqrt(var)
    x[x > x_mean + 6 * x_std] = (x_mean + 6 * x_std)[x > x_mean + 6 * x_std]
    x[x < x_mean - 6 * x_std] = (x_mean - 6 * x_std)[x < x_mean - 6 * x_std]

    x = x.astype(float)
    # 计算
    x_mean = ta.MA(x, timeperiod=window)
    sum_square = ta.MA(x ** 2, timeperiod=window) * window
    var = np.sqrt((sum_square - window * (x_mean ** 2)) / (window - 1))
    var[var <= 0] = 1e-8
    x_std = np.sqrt(var)
    return np.nan_to_num(x / x_std)
    return np.nan_to_num(x / x_std)


def scaler_mm_abs(x, window=500):
    """
     minmax_abs缩放
    :param x:
    :param window:
    :return:
    """
    # 滚动缩尾
    x = x.astype(float)
    window1 = 200
    x_mean = ta.MA(x, timeperiod=window1)
    sum_square = ta.MA(x ** 2, timeperiod=window1) * window1
    var = np.sqrt((sum_square - window1 * (x_mean ** 2)) / (window1 - 1))
    var[var <= 0] = 1e-8
    x_std = np.sqrt(var)
    x[x > x_mean + 6 * x_std] = (x_mean + 6 * x_std)[x > x_mean + 6 * x_std]
    x[x < x_mean - 6 * x_std] = (x_mean - 6 * x_std)[x < x_mean - 6 * x_std]
    x = x.astype(float)
    # 计算
    x_mmabs = np.maximum(np.abs(ta.MAX(x, timeperiod=window)), np.abs(ta.MIN(x, timeperiod=window)))
    return np.nan_to_num(x / x_mmabs)


def scaler_Lp(x, window=500):
    """
    L2范数缩放
    :param x:
    :param window:
    :return:
    """
    x = x.astype(float)
    # 滚动缩尾
    window1 = 200
    x_mean = ta.MA(x, timeperiod=window1)
    sum_square = ta.MA(x ** 2, timeperiod=window1) * window1
    var = np.sqrt((sum_square - window1 * (x_mean ** 2)) / (window1 - 1))
    var[var <= 0] = 1e-8
    x_std = np.sqrt(var)
    x[x > x_mean + 6 * x_std] = (x_mean + 6 * x_std)[x > x_mean + 6 * x_std]
    x[x < x_mean - 6 * x_std] = (x_mean - 6 * x_std)[x < x_mean - 6 * x_std]
    x = x.astype(float)
    # 计算
    p = 2
    x_L2 = (ta.MA(np.abs(x) ** p, timeperiod=window)) ** (1 / p)
    return np.nan_to_num(x / x_L2)

scaler = scaler_Lp


def _ta_ht_trendline(x1):
    x1 = x1.flatten()
    x = np.nan_to_num(talib.HT_TRENDLINE(x1))
    return scaler(x)

def _ta_ht_dcperiod(x1):
    x1 = x1.flatten()
    x = np.nan_to_num(talib.HT_DCPERIOD(x1))
    return scaler(x)

def _ta_obv(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.OBV(x1, x2))
    return scaler(x)

def _ts_cov_20(x1, x2):
    t = 20
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).cov(pd.Series(x2)))
    return scaler(x)

def _ts_cov_40(x1, x2):
    t = 40
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).cov(pd.Series(x2)))
    return scaler(x)

def _ts_corr_20(x1, x2):
    t = 20
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).corr(pd.Series(x2)))
    return scaler(x)

def _ts_corr_40(x1, x2):
    t = 40
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).corr(pd.Series(x2)))
    return scaler(x)

def _ts_sma_8(x1):  # the i_th element is the simple moving average of the elements in the n-period time series from the past
    t = 8
    x1 = x1.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).mean()))
    return scaler(x)

def _ts_sma_21(x1):  # the i_th element is the simple moving average of the elements in the n-period time series from the past
    t = 21
    x1 = x1.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).mean()))
    return scaler(x)

def _ts_sma_55(x1):  # the i_th element is the simple moving average of the elements in the n-period time series from the past
    t = 55
    x1 = x1.flatten()
    x = (np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).mean()))
    return scaler(x)

def _ts_wma_8(x1):  #修正于2023年6月22日，全向量化操作
    t = 8
    x1 = x1.flatten()
    weight_list = np.arange(1, t + 1)
    weight_list = weight_list / np.sum(weight_list)
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.dot(x, weight_list)
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_wma_21(x1):  #修正于2023年6月22日，全向量化操作
    t = 21
    x1 = x1.flatten()
    weight_list = np.arange(1, t + 1)
    weight_list = weight_list / np.sum(weight_list)
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.dot(x, weight_list)
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_wma_55(x1):  #修正于2023年6月22日，全向量化操作
    t = 55
    x1 = x1.flatten()
    weight_list = np.arange(1, t + 1)
    weight_list = weight_list / np.sum(weight_list)
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.dot(x, weight_list)
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)


def _ts_lag_3(x1):
    t = 3
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).shift(periods=t))
    return scaler(x)

def _ts_lag_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).shift(periods=t))
    return scaler(x)

def _ts_lag_17(x1):
    t = 17
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).shift(periods=t))
    return scaler(x)

def _ts_delta_3(x1):
    t = 3
    x1 = x1.flatten()
    x = np.nan_to_num(x1 - np.nan_to_num(pd.Series(x1).shift(periods=t)))
    return scaler(x)

def _ts_delta_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(x1 - np.nan_to_num(pd.Series(x1).shift(periods=t)))
    return scaler(x)

def _ts_delta_17(x1):
    t = 17
    x1 = x1.flatten()
    x = np.nan_to_num(x1 - np.nan_to_num(pd.Series(x1).shift(periods=t)))
    return scaler(x)

def _ts_sum_3(x1):  #修正于2023年6月22日，全向量化操作
    t = 3
    x1 = x1.flatten()
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.sum(x, axis=1)
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_sum_8(x1):  #修正于2023年6月22日，全向量化操作
    t = 8
    x1 = x1.flatten()
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.sum(x, axis=1)
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_sum_17(x1):  #修正于2023年6月22日，全向量化操作
    t = 17
    x1 = x1.flatten()
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.sum(x, axis=1)
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_prod_3(x1):  #修正于2023年6月22日，全向量化操作
    t = 3
    x1 = x1.flatten()
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.prod(x, axis=1)
    x = np.nan_to_num(np.concatenate(([0]*(t - 1), x)))
    return scaler(x)

def _ts_prod_8(x1):  #修正于2023年6月22日，全向量化操作
    t = 8
    x1 = x1.flatten()
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.prod(x, axis=1)
    x = np.nan_to_num(np.concatenate(([0]*(t - 1), x)))
    return scaler(x)

def _ts_prod_17(x1):  #修正于2023年6月22日，全向量化操作
    t = 17
    x1 = x1.flatten()
    x = sliding_window_view(pd.Series(x1), window_shape=t)
    x = np.prod(x, axis=1)
    x = np.nan_to_num(np.concatenate(([0]*(t - 1), x)))
    return scaler(x)

def _ts_std_10(x1):  # the i_th element is the standard deviation of the elements in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).std())
    return scaler(x)

def _ts_std_20(x1):  # the i_th element is the standard deviation of the elements in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).std())
    return scaler(x)

def _ts_std_40(x1):  # the i_th element is the standard deviation of the elements in the n-period time series from the past
    t = 40
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).std())
    return scaler(x)

def _ts_skew_10(x1):  # the i_th element is the skewness of the elements in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).skew())
    return scaler(x)

def _ts_skew_20(x1):  # the i_th element is the skewness of the elements in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).skew())
    return scaler(x)

def _ts_skew_40(x1):  # the i_th element is the skewness of the elements in the n-period time series from the past
    t = 40
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).skew())
    return scaler(x)

def _ts_kurt_10(x1):  # the i_th element is the kurtosis of the elements in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).kurt())
    return scaler(x)

def _ts_kurt_20(x1):  # the i_th element is the kurtosis of the elements in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).kurt())
    return scaler(x)

def _ts_kurt_40(x1):  # the i_th element is the kurtosis of the elements in the n-period time series from the past
    t = 40
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).kurt())
    return scaler(x)

def _ts_min_5(x1):  # the i_th element is the minimum value in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return scaler(x)

def _ts_min_10(x1):  # the i_th element is the minimum value in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return scaler(x)

def _ts_min_20(x1):  # the i_th element is the minimum value in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return scaler(x)

def _ts_max_5(x1):  # the i_th element is the maximum value in the n-period time series from the past
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max())
    return scaler(x)

def _ts_max_10(x1):  # the i_th element is the maximum value in the n-period time series from the past
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max())
    return scaler(x)

def _ts_max_20(x1):  # the i_th element is the maximum value in the n-period time series from the past
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max())
    return scaler(x)

def _ts_range_5(x1):
    t = 5
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max()) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return scaler(x)

def _ts_range_10(x1):
    t = 10
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max()) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return scaler(x)

def _ts_range_20(x1):
    t = 20
    x = np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).max()) - np.nan_to_num(pd.Series(x1).rolling(window=t, min_periods=int(t / 2)).min())
    return scaler(x)

def _ts_argmin_5(x1): #修正于2023年6月22日，全向量化操作
    t = 5
    x = np.argmin(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1 # 修正这一步
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argmin_10(x1): #修正于2023年6月22日，全向量化操作
    t = 10
    x = np.argmin(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1 # 修正这一步
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argmin_20(x1): #修正于2023年6月22日，全向量化操作
    t = 20
    x = np.argmin(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1 # 修正这一步
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argmax_5(x1):  #修正于2023年6月22日，全向量化操作
    t = 5
    x = np.argmax(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1 # 修正这一步
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argmax_10(x1): #修正于2023年6月22日，全向量化操作
    t = 10
    x = np.argmax(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1 # 修正这一步
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argmax_20(x1): #修正于2023年6月22日，全向量化操作
    t = 20
    x = np.argmax(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1 # 修正这一步
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argrange_5(x1): #修正于2023年6月22日，全向量化操作
    t = 5
    x = np.nan_to_num(np.argmax(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1) - np.nan_to_num(np.argmin(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1)
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argrange_10(x1): #修正于2023年6月22日，全向量化操作
    t = 10
    x = np.nan_to_num(np.argmax(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1) - np.nan_to_num(np.argmin(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1)
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_argrange_20(x1): #修正于2023年6月22日，全向量化操作
    t = 20
    x = np.nan_to_num(np.argmax(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1) - np.nan_to_num(np.argmin(sliding_window_view(x1.flatten(), window_shape=t), axis=1) + 1)
    x = np.nan_to_num(np.concatenate((np.array([0] * (t - 1)), x)))
    return scaler(x)

def _ts_rank_5(x1):  #修正于2023年6月22日，全向量化操作
    t = 5
    x1 = x1.flatten()
    arr=sliding_window_view(x1, t)
    x = percentileofscore(arr, arr[:, -1]) * 100
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_rank_10(x1):  #修正于2023年6月22日，全向量化操作
    t = 10
    x1 = x1.flatten()
    arr=sliding_window_view(x1, t)
    x = percentileofscore(arr, arr[:, -1]) * 100
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ts_rank_20(x1):  #修正于2023年6月22日，全向量化操作
    t = 20
    x1 = x1.flatten()
    arr=sliding_window_view(x1, t)
    x = percentileofscore(arr, arr[:, -1]) * 100
    x = np.nan_to_num(np.concatenate(([0] * (t - 1), x)))
    return scaler(x)

def _ta_beta_5(x1, x2):
    t = 5
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.BETA(x1, x2, timeperiod=t))
    return scaler(x)

def _ta_beta_10(x1, x2):
    t = 10
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.BETA(x1, x2, timeperiod=t))
    return scaler(x)

def _ta_beta_20(x1, x2):
    t = 20
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.BETA(x1, x2, timeperiod=t))
    return scaler(x)

def _ta_lr_slope_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_SLOPE(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_slope_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_SLOPE(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_slope_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_SLOPE(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_intercept_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_INTERCEPT(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_intercept_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_INTERCEPT(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_intercept_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_INTERCEPT(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_angle_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_ANGLE(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_angle_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_ANGLE(x1, timeperiod=t))
    return scaler(x)

def _ta_lr_angle_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.LINEARREG_ANGLE(x1, timeperiod=t))
    return scaler(x)

def _ta_tsf_5(x1):
    t = 5
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TSF(x1, timeperiod=t))
    return scaler(x)

def _ta_tsf_10(x1):
    t = 10
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TSF(x1, timeperiod=t))
    return scaler(x)

def _ta_tsf_20(x1):
    t = 20
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TSF(x1, timeperiod=t))
    return scaler(x)

def _ta_ema_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.EMA(x1, timeperiod=t))
    return scaler(x)

def _ta_ema_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.EMA(x1, timeperiod=t))
    return scaler(x)

def _ta_ema_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.EMA(x1, timeperiod=t))
    return scaler(x)

def _ta_dema_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.DEMA(x1, timeperiod=t))
    return scaler(x)

def _ta_dema_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.DEMA(x1, timeperiod=t))
    return scaler(x)

def _ta_dema_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.DEMA(x1, timeperiod=t))
    return scaler(x)

def _ta_kama_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.KAMA(x1, timeperiod=t))
    return scaler(x)

def _ta_kama_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.KAMA(x1, timeperiod=t))
    return scaler(x)

def _ta_kama_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.KAMA(x1, timeperiod=t))
    return scaler(x)

def _ta_tema_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TEMA(x1, timeperiod=t))
    return scaler(x)

def _ta_tema_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TEMA(x1, timeperiod=t))
    return scaler(x)

def _ta_tema_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TEMA(x1, timeperiod=t))
    return scaler(x)

def _ta_trima_8(x1):
    t = 8
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIMA(x1, timeperiod=t))
    return scaler(x)

def _ta_trima_21(x1):
    t = 21
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIMA(x1, timeperiod=t))
    return scaler(x)

def _ta_trima_55(x1):
    t = 55
    x1 = x1.flatten()
    x = np.nan_to_num(talib.TRIMA(x1, timeperiod=t))
    return scaler(x)

def _ta_rsi_6(x1):
    t = 6
    x1 = x1.flatten()
    x = np.nan_to_num(talib.RSI(x1, timeperiod=t))
    return scaler(x)

def _ta_rsi_12(x1):
    t = 12
    x1 = x1.flatten()
    x = np.nan_to_num(talib.RSI(x1, timeperiod=t))
    return scaler(x)

def _ta_rsi_24(x1):
    t = 24
    x1 = x1.flatten()
    x = np.nan_to_num(talib.RSI(x1, timeperiod=t))
    return scaler(x)

def _ta_cmo_14(x1):
    t = 14
    x1 = x1.flatten()
    x = np.nan_to_num(talib.CMO(x1, timeperiod=t))
    return scaler(x)

def _ta_cmo_25(x1):
    t = 25
    x1 = x1.flatten()
    x = np.nan_to_num(talib.CMO(x1, timeperiod=t))
    return scaler(x)

def _ta_adx_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADX(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_adx_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADX(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_adxr_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADXR(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_adxr_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.ADXR(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_aroonosc_14(x1, x2):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.AROONOSC(x1, x2, timeperiod=t))
    return scaler(x)

def _ta_aroonosc_25(x1, x2):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x = np.nan_to_num(talib.AROONOSC(x1, x2, timeperiod=t))
    return scaler(x)

def _ta_cci_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.CCI(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_cci_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.CCI(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_dx_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.DX(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_dx_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.DX(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_mfi_14(x1, x2, x3, x4):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = np.nan_to_num(talib.MFI(x1, x2, x3, x4, timeperiod=t))
    return scaler(x)

def _ta_mfi_25(x1, x2, x3, x4):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x4 = x4.flatten()
    x = np.nan_to_num(talib.MFI(x1, x2, x3, x4, timeperiod=t))
    return scaler(x)

def _ta_minus_di_14(x1, x2, x3):
    t = 14
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.MINUS_DI(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _ta_minus_di_25(x1, x2, x3):
    t = 25
    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()
    x = np.nan_to_num(talib.MINUS_DI(x1, x2, x3, timeperiod=t))
    return scaler(x)

def _diff(x1):
    """
    差分算子
    :param x1:
    :return:
    """
    x = np.zeros(x1.shape)
    x[1:] = np.diff(x1)
    return scaler(x)

def _macd(x1):
    x = np.nan_to_num(ta.MACD(x1)[2])
    return scaler(x)

def _ht_sine(x1):
    x = np.nan_to_num(ta.HT_SINE(x1)[0])
    return scaler(x)

def _ht_leadsine(x1):
    x = np.nan_to_num(ta.HT_SINE(x1)[1])
    return scaler(x)



add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
neg1 = _Function(function=np.negative, name='neg', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

# 增加部分的函数
tanh1 = _Function(function=_tanh, name='tanh', arity=1)
elu1 = _Function(function=_elu, name='elu', arity=1)
ta_ht_trendline = _Function(function=_ta_ht_trendline, name='TA_HT_TRENDLINE', arity=1)
ta_ht_dcperiod = _Function(function=_ta_ht_dcperiod, name='TA_HT_DCPERIOD', arity=1)
ta_obv = _Function(function=_ta_obv, name='TA_OBV', arity=2)

# 5月23日加入的因子：
ts_cov_20 = _Function(function=_ts_cov_20, name='TS_COV_20', arity=2)
ts_cov_40 = _Function(function=_ts_cov_40, name='TS_COV_40', arity=2)
ts_corr_20 = _Function(function=_ts_corr_20, name='TS_CORR_20', arity=2)
ts_corr_40 = _Function(function=_ts_corr_40, name='TS_CORR_40', arity=2)

# 11-19:
ts_sma_8 = _Function(function=_ts_sma_8, name='ts_sma_8', arity=1)
ts_sma_21 = _Function(function=_ts_sma_21, name='ts_sma_21', arity=1)
ts_sma_55 = _Function(function=_ts_sma_55, name='ts_sma_55', arity=1)
ts_wma_8 = _Function(function=_ts_wma_8, name='ts_wma_8', arity=1)
ts_wma_21 = _Function(function=_ts_wma_21, name='ts_wma_21', arity=1)
ts_wma_55 = _Function(function=_ts_wma_55, name='ts_wma_55', arity=1)
ts_lag_3 = _Function(function=_ts_lag_3, name='ts_lag_3', arity=1)
ts_lag_8 = _Function(function=_ts_lag_8, name='ts_lag_8', arity=1)
ts_lag_17 = _Function(function=_ts_lag_17, name='ts_lag_17', arity=1)

# 20-32
ts_delta_3 = _Function(function=_ts_delta_3, name='ts_delta_3', arity=1)
ts_delta_8 = _Function(function=_ts_delta_8, name='ts_delta_8', arity=1)
ts_delta_17 = _Function(function=_ts_delta_17, name='ts_delta_17', arity=1)
ts_sum_3 = _Function(function=_ts_sum_3, name='ts_sum_3', arity=1)
ts_sum_8 = _Function(function=_ts_sum_8, name='ts_sum_8', arity=1)
ts_sum_17 = _Function(function=_ts_sum_17, name='ts_sum_17', arity=1)
ts_prod_3 = _Function(function=_ts_prod_3, name='ts_prod_3', arity=1)
ts_prod_8 = _Function(function=_ts_prod_8, name='ts_prod_8', arity=1)
ts_prod_17 = _Function(function=_ts_prod_17, name='ts_prod_17', arity=1)
ts_std_10 = _Function(function=_ts_std_10, name='ts_std_10', arity=1)
ts_std_20 = _Function(function=_ts_std_20, name='ts_std_20', arity=1)
ts_std_40 = _Function(function=_ts_std_40, name='ts_std_40', arity=1)

# 33-44
ts_skew_10 = _Function(function=_ts_skew_10, name='ts_skew_10', arity=1)
ts_skew_20 = _Function(function=_ts_skew_20, name='ts_skew_20', arity=1)
ts_skew_40 = _Function(function=_ts_skew_40, name='ts_skew_40', arity=1)
ts_kurt_10 = _Function(function=_ts_kurt_10, name='ts_kurt_10', arity=1)
ts_kurt_20 = _Function(function=_ts_kurt_20, name='ts_kurt_20', arity=1)
ts_kurt_40 = _Function(function=_ts_kurt_40, name='ts_kurt_40', arity=1)
ts_min_5 = _Function(function=_ts_min_5, name='ts_min_5', arity=1)
ts_min_10 = _Function(function=_ts_min_10, name='ts_min_10', arity=1)
ts_min_20 = _Function(function=_ts_min_20, name='ts_min_20', arity=1)
ts_max_5 = _Function(function=_ts_max_5, name='ts_max_5', arity=1)
ts_max_10 = _Function(function=_ts_max_10, name='ts_max_10', arity=1)
ts_max_20 = _Function(function=_ts_max_20, name='ts_max_20', arity=1)

# 45-56
ts_range_5 = _Function(function=_ts_range_5, name='ts_range_5', arity=1)
ts_range_10 = _Function(function=_ts_range_10, name='ts_range_10', arity=1)
ts_range_20 = _Function(function=_ts_range_20, name='ts_range_20', arity=1)
ts_argmin_5 = _Function(function=_ts_argmin_5, name='ts_argmin_5', arity=1)
ts_argmin_10 = _Function(function=_ts_argmin_10, name='ts_argmin_10', arity=1)
ts_argmin_20 = _Function(function=_ts_argmin_20, name='ts_argmin_20', arity=1)
ts_argmax_5 = _Function(function=_ts_argmax_5, name='ts_argmax_5', arity=1)
ts_argmax_10 = _Function(function=_ts_argmax_10, name='ts_argmax_10', arity=1)
ts_argmax_20 = _Function(function=_ts_argmax_20, name='ts_argmax_20', arity=1)
ts_argrange_5 = _Function(function=_ts_argrange_5, name='ts_argrange_5', arity=1)
ts_argrange_10 = _Function(function=_ts_argrange_10, name='ts_argrange_10', arity=1)
ts_argrange_20 = _Function(function=_ts_argrange_20, name='ts_argrange_20', arity=1)

# 57-68
ts_rank_5 = _Function(function=_ts_rank_5, name='ts_rank_5', arity=1)
ts_rank_10 = _Function(function=_ts_rank_10, name='ts_rank_10', arity=1)
ts_rank_20 = _Function(function=_ts_rank_20, name='ts_rank_20', arity=1)
ta_beta_5 = _Function(function=_ta_beta_5, name='ta_beta_5', arity=2)
ta_beta_10 = _Function(function=_ta_beta_10, name='ta_beta_10', arity=2)
ta_beta_20 = _Function(function=_ta_beta_20, name='ta_beta_20', arity=2)
ta_lr_slope_5 = _Function(function=_ta_lr_slope_5, name='ta_lr_slope_5', arity=1)
ta_lr_slope_10 = _Function(function=_ta_lr_slope_10, name='ta_lr_slope_10', arity=1)
ta_lr_slope_20 = _Function(function=_ta_lr_slope_20, name='ta_lr_slope_20', arity=1)

# 69-80
ta_lr_intercept_5 = _Function(function=_ta_lr_intercept_5, name='ta_lr_intercept_5', arity=1)
ta_lr_intercept_10 = _Function(function=_ta_lr_intercept_10, name='ta_lr_intercept_10', arity=1)
ta_lr_intercept_20 = _Function(function=_ta_lr_intercept_20, name='ta_lr_intercept_20', arity=1)
ta_lr_angle_5 = _Function(function=_ta_lr_angle_5, name='ta_lr_angle_5', arity=1)
ta_lr_angle_10 = _Function(function=_ta_lr_angle_10, name='ta_lr_angle_10', arity=1)
ta_lr_angle_20 = _Function(function=_ta_lr_angle_20, name='ta_lr_angle_20', arity=1)
ta_tsf_5 = _Function(function=_ta_tsf_5, name='ta_tsf_5', arity=1)
ta_tsf_10 = _Function(function=_ta_tsf_10, name='ta_tsf_10', arity=1)
ta_tsf_20 = _Function(function=_ta_tsf_20, name='ta_tsf_20', arity=1)
ta_ema_8 = _Function(function=_ta_ema_8, name='ta_ema_8', arity=1)
ta_ema_21 = _Function(function=_ta_ema_21, name='ta_ema_21', arity=1)
ta_ema_55 = _Function(function=_ta_ema_55, name='ta_ema_55', arity=1)

# 81-92
ta_dema_8 = _Function(function=_ta_dema_8, name='ta_dema_8', arity=1)
ta_dema_21 = _Function(function=_ta_dema_21, name='ta_dema_21', arity=1)
ta_dema_55 = _Function(function=_ta_dema_55, name='ta_dema_55', arity=1)
ta_kama_8 = _Function(function=_ta_kama_8, name='ta_dema_8', arity=1)
ta_kama_21 = _Function(function=_ta_kama_21, name='ta_dema_21', arity=1)
ta_kama_55 = _Function(function=_ta_kama_55, name='ta_dema_55', arity=1)
ta_tema_8 = _Function(function=_ta_tema_8, name='ta_tema_8', arity=1)
ta_tema_21 = _Function(function=_ta_tema_21, name='ta_tema_21', arity=1)
ta_tema_55 = _Function(function=_ta_tema_55, name='ta_tema_55', arity=1)
ta_trima_8 = _Function(function=_ta_trima_8, name='ta_trima_8', arity=1)
ta_trima_21 = _Function(function=_ta_trima_21, name='ta_trima_21', arity=1)
ta_trima_55 = _Function(function=_ta_trima_55, name='ta_trima_55', arity=1)

# 93-105
ta_rsi_6 = _Function(function=_ta_rsi_6, name='ta_rsi_6', arity=1)
ta_rsi_12 = _Function(function=_ta_rsi_12, name='ta_rsi_12', arity=1)
ta_rsi_24 = _Function(function=_ta_rsi_24, name='ta_rsi_24', arity=1)
ta_cmo_14 = _Function(function=_ta_cmo_14, name='ta_cmo_14', arity=1)
ta_cmo_25 = _Function(function=_ta_cmo_25, name='ta_cmo_25', arity=1)
ta_adx_14 = _Function(function=_ta_adx_14, name='ta_adx_14', arity=3)
ta_adx_25 = _Function(function=_ta_adx_25, name='ta_adx_25', arity=3)
ta_adxr_14 = _Function(function=_ta_adxr_14, name='ta_adxr_14', arity=3)
ta_adxr_25 = _Function(function=_ta_adxr_25, name='ta_adxr_25', arity=3)
ta_aroonosc_14 = _Function(function=_ta_aroonosc_14, name='ta_aroonosc_14', arity=2)
ta_aroonosc_25 = _Function(function=_ta_aroonosc_25, name='ta_aroonosc_25', arity=2)
ta_cci_14 = _Function(function=_ta_cci_14, name='ta_cci_14', arity=3)
ta_cci_25 = _Function(function=_ta_cci_25, name='ta_cci_25', arity=3)

# 117-
ta_dx_14 = _Function(function=_ta_dx_14, name='ta_dx_14', arity=3)
ta_dx_25 = _Function(function=_ta_dx_25, name='ta_dx_25', arity=3)
ta_mfi_14 = _Function(function=_ta_mfi_14, name='ta_mfi_14', arity=4)
ta_mfi_25 = _Function(function=_ta_mfi_25, name='ta_mfi_25', arity=4)
ta_minus_di_14 = _Function(function=_ta_minus_di_14, name='ta_minus_di_14', arity=3)
ta_minus_di_25 = _Function(function=_ta_minus_di_25, name='ta_minus_di_25', arity=3)


# 2024-07-06新添加

diff = _Function(function=_diff, name='diff', arity=1)
macd = _Function(function=_macd, name='macd', arity=1)
ht_sine = _Function(function=_ht_sine, name='ht_sine', arity=1)
ht_leadsine = _Function(function=_ht_leadsine, name='ht_leadsine', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'abs': abs1,
                 'neg': neg1,
                #  'max': max2,
                #  'min': min2,
                #  'sin': sin1,
                #  'cos': cos1,
                 
                 # 下面对应的是增加部分
                 'tanh': tanh1,
                 'elu': elu1,
                 'TA_HT_TRENDLINE': ta_ht_trendline,
                 'TA_HT_DCPERIOD': ta_ht_dcperiod,
                 'TA_OBV': ta_obv,
                 # 5月23日加入
                 # 1-10：
                 'TS_COV_20' : ts_cov_20,
                 'TS_COV_40' : ts_cov_40,
                 'TS_CORR_20' : ts_corr_20,
                 'TS_CORR_40' : ts_corr_40,
                 # 11-19:
                 'ts_sma_8' : ts_sma_8,
                 'ts_sma_21' : ts_sma_21,
                 'ts_sma_55' : ts_sma_55,
                 'ts_wma_8' : ts_wma_8,
                 'ts_wma_21' : ts_wma_21,
                 'ts_wma_55' : ts_wma_55,
                 'ts_lag_3' : ts_lag_3,
                 'ts_lag_8' : ts_lag_8,
                 'ts_lag_17' : ts_lag_17,
                 # 20-32
                 'ts_delta_3' : ts_delta_3,
                 'ts_delta_8' : ts_delta_8,
                 'ts_delta_17' : ts_delta_17,
                 'ts_sum_3' : ts_sum_3,
                 'ts_sum_8' : ts_sum_8,
                 'ts_sum_17' : ts_sum_17,
                 'ts_prod_3' : ts_prod_3,
                 'ts_prod_8' : ts_prod_8,
                 'ts_prod_17' : ts_prod_17,
                 'ts_std_10' : ts_std_10,
                 'ts_std_20' : ts_std_20,
                 'ts_std_40' : ts_std_40,
                 # 33-44 
                 'ts_skew_10' : ts_skew_10,
                 'ts_skew_20' : ts_skew_20,
                 'ts_skew_40' : ts_skew_40,
                 'ts_kurt_10' : ts_kurt_10,
                 'ts_kurt_20' : ts_kurt_20,
                 'ts_kurt_40' : ts_kurt_40,
                 'ts_min_5' : ts_min_5,
                 'ts_min_10' : ts_min_10,
                 'ts_min_20' : ts_min_20,
                 'ts_max_5' : ts_max_5,
                 'ts_max_10' : ts_max_10,
                 'ts_max_20' : ts_max_20,
                 # 45-56
                 'ts_range_5' : ts_range_5,
                 'ts_range_10' : ts_range_10,
                 'ts_range_20' : ts_range_20,
                 'ts_argmin_5' : ts_argmin_5,
                 'ts_argmin_10' : ts_argmin_10,
                 'ts_argmin_20' : ts_argmin_20,
                 'ts_argmax_5' : ts_argmax_5,
                 'ts_argmax_10' : ts_argmax_10,
                 'ts_argmax_20' : ts_argmax_20,
                 'ts_argrange_5' : ts_argrange_5,
                 'ts_argrange_10' : ts_argrange_10,
                 'ts_argrange_20' : ts_argrange_20,
                 # 57-68
                 'ts_rank_5' : ts_rank_5,
                 'ts_rank_10' : ts_rank_10,
                 'ts_rank_20' : ts_rank_20,
                 'ta_beta_5' : ta_beta_5,
                 'ta_beta_10' : ta_beta_10,
                 'ta_beta_20' : ta_beta_20,
                 'ta_lr_slope_5' : ta_lr_slope_5,
                 'ta_lr_slope_10' : ta_lr_slope_10,
                 'ta_lr_slope_20' : ta_lr_slope_20,
                 # 69-80
                 'ta_lr_intercept_5' : ta_lr_intercept_5,
                 'ta_lr_intercept_10' : ta_lr_intercept_10,
                 'ta_lr_intercept_20' : ta_lr_intercept_20,
                 'ta_lr_angle_5' : ta_lr_angle_5,
                 'ta_lr_angle_10' : ta_lr_angle_10,
                 'ta_lr_angle_20' : ta_lr_angle_20,
                 'ta_tsf_5' : ta_tsf_5,
                 'ta_tsf_10' : ta_tsf_10,
                 'ta_tsf_20' : ta_tsf_20,
                 'ta_ema_8' : ta_ema_8,
                 'ta_ema_21' : ta_ema_21,
                 'ta_ema_55' : ta_ema_55,
                 # 81-92
                 'ta_dema_8': ta_dema_8,
                 'ta_dema_21': ta_dema_21,
                 'ta_dema_55': ta_dema_55,
                 'ta_kama_8' : ta_kama_8,
                 'ta_kama_21' : ta_kama_21,
                 'ta_kama_55' : ta_kama_55,
                 'ta_tema_8' : ta_tema_8,
                 'ta_tema_21' : ta_tema_21,
                 'ta_tema_55' : ta_tema_55,
                 'ta_trima_8' : ta_trima_8,
                 'ta_trima_21' : ta_trima_21,
                 'ta_trima_55' : ta_trima_55,
                 # 93-105
                 'ta_rsi_6' : ta_rsi_6,
                 'ta_rsi_12' : ta_rsi_12,
                 'ta_rsi_24' : ta_rsi_24,
                 'ta_cmo_14' : ta_cmo_14,
                 'ta_cmo_25' : ta_cmo_25,
                 'ta_adx_14' : ta_adx_14,
                 'ta_adx_25' : ta_adx_25,
                 'ta_adxr_14' : ta_adxr_14,
                 'ta_adxr_25' : ta_adxr_25,
                 'ta_aroonosc_14' : ta_aroonosc_14,
                 'ta_aroonosc_25' : ta_aroonosc_25,
                 'ta_cci_14' : ta_cci_14,
                 'ta_cci_25' : ta_cci_25,
                 # 117-120
                 'ta_dx_14' : ta_dx_14,
                 'ta_dx_25' : ta_dx_25,
                 'ta_mfi_14' : ta_mfi_14,
                 'ta_mfi_25' : ta_mfi_25,
                 'ta_minus_di_14' : ta_minus_di_14,
                 'ta_minus_di_25' : ta_minus_di_25,
                 
                 # 2024-07-06 添加四个函数
                 'diff': diff,
                 'macd': macd,
                 'ht_sine': ht_sine,
                 'ht_leadsine': ht_leadsine,
                 }
