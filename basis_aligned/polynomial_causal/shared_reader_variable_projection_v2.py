"""Preserve gradient recording when physical() evaluates an uncached point."""
import numpy as np
from shared_reader_variable_projection_v1 import Objective as BaseObjective


class Objective(BaseObjective):
    def physical(self,point):
        if self.last_point is None or not np.array_equal(point,self.last_point):
            self.evaluate(point)
        return super().physical(point)
