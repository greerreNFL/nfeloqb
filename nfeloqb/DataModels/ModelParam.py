from dataclasses import dataclass, asdict

@dataclass
class ModelParam:
  '''
  Dataclass for a model parameter, which compose the ModelConfig.
  '''
  param_name: str
  value: float
  description: str
  opti_min: float
  opti_max: float

  @classmethod
  def from_dict(cls, dict_data):
    '''
    Init a ModelParam from a dict object.
    '''
    return cls(**dict_data)
  
  def asdict(self):
    '''
    Return a dict representation of the ModelParam.
    '''
    return asdict(self)
  
  def as_config_dict(self):
    '''
    Returns the param in a format for the config json file.
    '''
    return {
      'value': self.value,
      'description': self.description,
      'opti_min': self.opti_min,
      'opti_max': self.opti_max
    }
