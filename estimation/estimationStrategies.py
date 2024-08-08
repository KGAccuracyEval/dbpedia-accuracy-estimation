import numpy as np

from scipy import stats


class TWCSEstimator(object):

    def __init__(self, alpha=0.05):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha / 2)

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on the sample

        :param sample: input sample (i.e., clusters of triples) used for estimation
        :return: KG accuracy estimate
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster) / len(cluster) for cluster in sample]
        # compute estimate
        return sum(cae) / len(cae)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster) / len(cluster) for cluster in sample]
        # compute estimate
        ae = sum(cae) / len(cae)

        # count number of clusters in sample
        n = len(sample)

        if n * (n - 1) != 0:  # compute variance
            var = (1 / (n * (n - 1))) * sum([(cae[i] - ae) ** 2 for i in range(n)])
        else:  # set variance to inf
            var = np.inf
        return var

    def computeMoE(self, var):
        """
        Compute the Margin of Error (MoE) based on the sample and the Normal critical value z with right-tail probability α/2

        :param var: variance
        :return: the MoE value
        """

        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        return moe


class STWCSEstimator(object):
    def __init__(self, alpha=0.05):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

    def estimate(self, strataEstimates, strataWeights):
        """
        Estimate the KG accuracy based on the sample

        :param strataEstimates: strata estimates
        :param strataWeights: strata weights
        :return: KG accuracy estimate
        """

        # compute estimate
        return sum([acc * weight for acc, weight in zip(strataEstimates, strataWeights)])

    def computeVar(self, strataVars, strataWeights):
        """
        Compute the sample variance

        :param strataVars: strata variances
        :param strataWeights: strata weights
        :return: sample standard deviation
        """

        # compute variance
        return sum([var * (weight ** 2) for var, weight in zip(strataVars, strataWeights)])

    def computeMoE(self, var):
        """
        Compute the Margin of Error (MoE) based on the sample and the Normal critical value z with right-tail probability α/2

        :param var: variance
        :return: the MoE value
        """

        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        return moe
