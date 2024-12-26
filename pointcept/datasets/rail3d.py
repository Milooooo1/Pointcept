"""
Rail3D Dataset

Author: Milo Beliën (milo.belien@sweco.nl)
"""

import os
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Rail3DDataset(DefaultDataset):
    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)]
