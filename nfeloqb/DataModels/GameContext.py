## Built-in ##
from dataclasses import dataclass, field, asdict
from typing import Optional

## Packages ##
import pandas as pd

## Models ##
from .ModelConfig import ModelConfig

## utilities ##
from .Utilities import s_curve

@dataclass
class GameContext:
    '''
    An object that contains state about a quarterback
    '''
    ## initing meta ##
    game_id: str
    config: ModelConfig
    ## optional ##
    temp: Optional[int] = None
    wind: Optional[int] = None
    ## post init ##
    params: dict = field(init=False)

    def __post_init__(self):
        '''
        convenience method to unpack the config
        '''
        ## unpack the config for convenience ##
        self.params = self.config.values
    
    def weather_adj(self) -> float:
        '''
        Calculate the negative adjustment for wind and temp
        '''
        ## handle values ##
        wind = max(0, min(30, self.wind-5 if not pd.isnull(self.wind) else 0))
        temp = max(0, self.temp if not pd.isnull(self.temp) else 70)
        ## calc adjs ##
        wind_adj = s_curve(
            self.params['wind_disc_height'],
            self.params['wind_disc_mp'],
            wind,
            'up'
        )
        temp_adj = s_curve(
            self.params['temp_disc_height'],
            self.params['temp_disc_mp'],
            temp,
            'down'
        )
        ## calc the adjustment ##
        return temp_adj + wind_adj
