""" Panel classification Interface """

from collections import OrderedDict
import json
import numpy as np


class PanelClasses():
    """ Interface to access panel classification by role """
    def __init__(self, classes_file):

        self.filename = classes_file

        with open(classes_file, 'r') as f:
            # preserve the order of classes names
            self.classes = json.load(f, object_pairs_hook=OrderedDict)

        self.names = list(self.classes.keys())
        
        self.panel_to_idx = {}
        for idx, class_name in enumerate(self.classes):
            self.panel_to_idx[class_name] = idx
            for key in self.classes[class_name]:
                self.panel_to_idx[key] = idx
        
    def __len__(self):
        return len(self.classes)

    def class_idx(self, panel):
        """
            Return idx of class for given panel (name) from given template(name)
        """
        return self.panel_to_idx[panel]

    def class_name(self, idx):
        return self.names[idx]

    def map(self, panel_list):
        """Map the list of panels for given template to the given list"""
        
        out_list = np.empty(len(panel_list))
        for idx, panel in enumerate(panel_list):
            if panel == 'stitch':
                out_list[idx] = -1
                print(f'{self.__class__.__name__}::Warning::Mapping stitch label')
            else:
                out_list[idx] = self.panel_to_idx[panel]

        return out_list
