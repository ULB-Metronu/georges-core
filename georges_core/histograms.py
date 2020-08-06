import boost_histogram as bh


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