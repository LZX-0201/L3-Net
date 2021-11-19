from model.model_base import ModelBase
import factory.model_factory as ModelFactory
import torch.nn as nn
import torch.nn.functional as F
import torch


class SceneBoxEncoder(ModelBase):
    def __init__(self, config):
        super(SceneBoxEncoder, self).__init__()
        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 200)


    def expand_boxes(self, boxes):
        box_vec = []
        for i in boxes:
            boxlist = [box for box in i]
            box_vec.append(torch.cat(boxlist, dim=0))
        return torch.stack(box_vec)

    def forward(self, data):
        x = self.expand_boxes(data['boxes'])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        batch_list = []
        for frm in x:
            pieces = frm.chunk(25, dim=0)
            pieces = [i.unsqueeze(0) for i in pieces]
            pieces = torch.cat(pieces, dim=0).unsqueeze(0)
            batch_list.append(pieces)
        return torch.cat(batch_list, dim=0)



    @staticmethod
    def check_config(config):
        pass

    @staticmethod
    def build_module(config):
        SceneBoxEncoder.check_config(config)
        return SceneBoxEncoder()
