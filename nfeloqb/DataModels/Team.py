from dataclasses import dataclass, field
from typing import Optional

from .ModelConfig import ModelConfig

@dataclass
class Team:
    '''
    Object representing a team
    '''
    ## initing meta ##
    abbr: str
    config: ModelConfig
    ## state ##
    last_game_date_off: Optional[str] = field(default=None, init=False)
    last_game_season_off: Optional[int] = field(default=None, init=False)
    last_game_date_def: Optional[str] = field(default=None, init=False)
    last_game_season_def: Optional[int] = field(default=None, init=False)
    off_value: float = field(init=False)
    def_value: float = field(init=False)
    season_adjs: float = field(default=0, init=False)
    ## extra ##
    params: dict = field(init=False)

    def __post_init__(self):
        ## init the values ##
        ## unpack params from config for convenience ##
        self.params = self.config.values    
        ## init the values ##
        self.off_value = self.params['init_value']
        self.def_value = 0 ## def is a relative measure, so it is initialized to 0
    
    def update_off_value(self,
        value: float,
        qb_adj: float,
        gameday: str,
        season: int
    ) -> None:
        '''
        Update the team's offensive value after a game

        Parameters:
        * value: The value of the team's offensive performance, with def adjusted
        * qb_adj: The adjustment made to the starting QB's value
        * gameday: The date of the game
        * season: The season of the game

        Returns:
        * None
        '''
        ## update the value ##
        self.off_value = (
            self.params['team_off_sf'] * value +
            (1 - self.params['team_off_sf']) * self.off_value
        )
        ## track the cumulative qb adj ##
        self.season_adjs += qb_adj
        ## update the last game date ##
        self.last_game_date_off = gameday
        self.last_game_season_off = season
    
    def update_def_value(self,
        value: float,
        gameday: str,
        season: int
    ) -> None:
        '''
        Update the team's defensive value after a game

        Parameters:
        * value: The value of the team's defensive performance, which is measured as the difference
        between the opposing QB's game value, and their expected value before adjusting for defense
        * gameday: The date of the game
        * season: The season of the game

        Returns:
        * None
        '''
        self.def_value = (
            self.params['team_def_sf'] * value +
            (1 - self.params['team_def_sf']) * self.def_value
        )
        ## update the last game date ##
        self.last_game_date_def = gameday
        self.last_game_season_def = season
    
    def regress_offense(self,
        qb_val: float,
        prev_season_league_avg: float
    ) -> None:
        '''
        Handle the offseason regression of a team's offensive value

        Parameters:
        * qb_val: The value of the team's Week 1 starting QB
        * prev_season_league_avg: The league average value of QBs from the previous season

        Returns:
        * None
        '''
        ## normalize the regression coefficients so they are not greater than 1 ##  
        total_regression = self.params['team_off_qb_reversion'] + self.params['team_off_league_reversion']
        if total_regression > 1:
            self.params['team_off_qb_reversion'] = self.params['team_off_qb_reversion'] / total_regression
            self.params['team_off_league_reversion'] = self.params['team_off_league_reversion'] / total_regression
        ## regress ##
        self.off_value = (
            (
                1 -
                self.params['team_off_qb_reversion'] -
                self.params['team_off_league_reversion']
            ) * self.off_value +
            self.params['team_off_qb_reversion'] * qb_val +
            self.params['team_off_league_reversion'] * prev_season_league_avg
        )
        ## reset the cumulative qb adj ##
        self.season_adjs = 0
        
    def regress_defense(self) -> None:
        '''
        Handle the offseason regression of a team's defensive value. Team defensesive value
        is a measure of how much better or worse a QB performs relative to expectation against
        a defense. Thus, an average defense would be 0.

        As a result, team's are regressed to 0 each offseason rather than a league average for defensive
        adjustments, which, due to variance, might be more or less than 0.

        Parameters:
        * None

        Returns:
        * None
        '''
        self.def_value = (
            (1 - self.params['team_def_reversion']) * self.def_value +
            self.params['team_def_reversion'] * 0
        )