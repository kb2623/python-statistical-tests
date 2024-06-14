# encoding=utf8

from typing import List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.lines as lines


class BasicStatistics:
	Name = ['BasicStatistics']

	def __init__(self, array: np.ndarray):
		self.array = array if isinstance(array, np.ndarray) else np.asarray(array)

	def min_value(self) -> Union[int, float, Any]: return self.array.min()

	def max_value(self) -> Union[int, float, Any]: return self.array.max()

	def mean(self) -> Union[float, Any]: return self.array.mean()

	def median(self) -> Union[int, float, Any]: return np.median(self.array)

	def standard_deviation(self) -> float: return self.array.std(ddof=1)

	def generate_standard_report(self) -> str: return "Min: {0}, Max: {1}, Mean: {2}, Median: {3}, Std. {4}".format(self.min_value(), self.max_value(), self.mean(), self.median(), self.standard_deviation())


def wilcoxonSignedRanks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
	r"""Get rank values from signed wilcoxon test.

	Args:
		a: First data.
		b: Second data.

	Returns:
		1. Positive ranks.
		2. Negative ranks.
		3. T value
	"""
	y = a - b
	y_diff = y[y != 0]
	r = stats.rankdata(np.abs(y_diff))
	r_all = np.sum(r) / 2
	r_p, r_n = r_all + np.sum(r[np.where(y_diff > 0)]), r_all + np.sum(r[np.where(y_diff < 0)])
	return r_p, r_n, np.min([r_p, r_n])


def friedmanRanks(*arrs: List[np.ndarray]) -> np.ndarray:
	r = np.asarray([stats.rankdata([arrs[j][i] for j in range(len(arrs))]) for i in range(len(arrs[0]))])
	return np.asarray([np.sum(r[:, i]) / len(arrs[0]) for i in range(len(arrs))])


def cd(alpha: float, k: float, n: float) -> float:
	r"""Get critial distance for friedman test.

	Args:
		alpha: Fold value.
		k: Number of algorithms.
		n: Number of algorithm results.
	"""
	nemenyi_df = pd.read_csv('nemenyi.csv')
	q_a = nemenyi_df['%.2f' % alpha][nemenyi_df['k'] == k].values
	return q_a[0] * np.sqrt((k * (k + 1)) / (6 * n))


def nemenyiTest(data: np.ndarray, names: Optional[np.ndarray] = None, q: float = .05, s: float = .1, ax: Optional[mpl.axes.Axes] = None, ylabel: str = 'Average rank', xlabel: str = 'Algorithm') -> mpl.axes.Axes:
	r"""Plot Friedman Nemenyi plot.

	Args:
		data: TODO.
		names: TODO.
		q: TODO.
		s: Scaling factor.
		ax: TODO.

	Returns:
		Figure axes.
	"""
	cd_h = cd(q, len(data), len(data[0])) / 2.0
	r = friedmanRanks(*data)
	if not ax: f, ax = plt.subplots(figsize=(10, 10))
	if not names: names = np.arange(len(data))
	for i, e in enumerate(r): ax.errorbar(i, e, cd_h, fmt='o', linewidth=2, capsize=6, label=names[i])
	ax.set_xticklabels(['', *names])
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True, which='both', axis='y', linestyle='--')
	return ax


def wilcoxonTest(data: np.ndarray, names: np.ndarray, q: Optional[float] = None) -> pd.DataFrame:
	r"""Get p-values or tagged differences bettwen algorithms.

	Args:
		data: Multi dimensional array with algorithms data.
		names: Names of algorithms
		q: TODO.

	Returns:
		Dataframe with p-values or tagged differences.
	"""
	df = pd.DataFrame(np.asarray([[stats.wilcoxon(data[j], data[i])[1] if j != i else 1 for i in range(len(data))] for j in range(len(data))]), index=names, columns=names)
	if q is not None:
		for i in range(df.shape[0]):
			for j in range(df.shape[1]): df.iloc[i, j] = '+' if df.iloc[i, j] <= q else '-'
	return df


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
