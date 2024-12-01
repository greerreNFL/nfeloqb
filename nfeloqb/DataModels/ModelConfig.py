from dataclasses import dataclass

@dataclass
class ModelConfig:
  '''
  Typed object for the model config json file.
  '''
  player_regression_league_mp: float
  player_regression_league_height: float
  player_regression_career_mp: float
  player_regression_career_height: float
  player_sf: float
  player_career_sf_base: float
  player_career_sf_height: float
  player_career_sf_mp: float
  rookie_draft_intercept: float
  rookie_draft_slope: float
  rookie_league_reg: float
  rookie_league_cap: float
  rookie_undrafted_draft_number: float
  team_off_league_reversion: float
  team_off_qb_reversion: float
  team_off_sf: float
  team_def_sf: float
  team_def_reversion: float
  init_value: float

  @classmethod
  def from_dict(cls, json_data: dict) -> 'ModelConfig':
    '''
    Create a ModelConfig from a dictionary.
    '''
    ## validate the data ##
    missing_fields = []
    for field in fields(cls):
      if field.name not in json_data:
        missing_fields.append(field.name)
    if missing_fields:
      raise ValueError(f"Config missing required fields: {missing_fields}")
    ## create the object ##
    return cls(**json_data)

  @classmethod
  def from_file(cls, file_path: str) -> 'ModelConfig':
    with open(file_path, 'r') as file:
      json_data = json.load(file)
    return cls.from_dict(json_data)
