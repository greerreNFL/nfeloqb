## built ins ##
import json
from dataclasses import dataclass
import pathlib

## local data models ##
from .ModelParam import ModelParam

@dataclass
class ModelConfig:
  '''
  Class for a model config, which is comprised of ModelParam objects.
  '''
  params: dict[str, ModelParam]
  values: dict[str, float]

  @classmethod
  def from_dict(cls, dict_data: dict) -> 'ModelConfig':
    '''
    Create a ModelConfig from a dictionary.
    '''
    ## create params ##
    params = {}
    values = {}
    for k, v in dict_data.items():
      v['param_name'] = k
      param = ModelParam.from_dict(v)
      params[k] = param
      values[k] = param.value
    ## create the object ##
    return cls(params=params, values=values)

  @classmethod
  def from_file(cls, file_path: str) -> 'ModelConfig':
    '''
    Create a ModelConfig from a json file.
    '''
    with open(file_path, 'r') as fp:
      json_data = json.load(fp)
    return cls.from_dict(json_data)
  
  def update_config(self, new_values: dict[str, float]) -> None:
    '''
    Update the ModelConfig with new values.
    '''
    for k, v in new_values.items():
      self.values[k] = v
      self.params[k].value = v
  
  def to_config_dict(self) -> dict:
    '''
    Convert the ModelConfig to a config dictionary.
    '''
    return {k: v.as_config_dict() for k, v in self.params.items()}
  
  def to_file(self) -> None:
    '''
    Convert the ModelConfig to a dictionary and save it to a json file.
    '''
    ## get root of pacakge ##
    root_fp = pathlib.Path(__file__).parent.parent.parent.resolve()
    config_fp = '{0}/model_config.json'.format(root_fp)
    with open(config_fp, 'w') as file:
      json.dump(self.to_config_dict(), file, indent=4)
