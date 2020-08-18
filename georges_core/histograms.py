import boost_histogram as bh
import pandas as pd
from scipy.interpolate import interp1d


class ExtendedHistogram(bh.Histogram):
    @classmethod
    def from_pyboost(cls, bh):
        return cls(bh)

    def project_to_3d(self, weights=1):
        histo3d = bh.Histogram(*self.axes[0:3])

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    tmp = self[x, y, z, :].view() * weights
                    histo3d[x, y, z] = tmp.sum()

        return histo3d

    def compute_h10(self, conversionFactorFile):
        data = pd.read_table(conversionFactorFile, names=["energy", "h10_coeff"])
        f = interp1d(data['energy'].values, data['h10_coeff'].values)
        return self.project_to_3d(weights=f(self.axes[3].centers))