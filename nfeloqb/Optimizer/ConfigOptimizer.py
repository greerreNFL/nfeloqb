## built-in packages ##
from typing import Tuple, Any, Optional
import pathlib
import time
import datetime

## external packages ##
import pandas as pd
from scipy.optimize import minimize

## data models ##
from ..DataModels import ModelConfig, ModelParam
from ..Resources import QBModel

class ConfigOptimizer:
  '''
  Optimizer that returns the optimal value for each parameter in the model config.
  '''
  def __init__(self,
    data: pd.DataFrame,
    config: ModelConfig,
    tol=0.000001,
    step=0.00001,
    method='SLSQP',
    subset: list[str] = [],
    subset_name: str = 'subset'
  ):
    self.data: pd.DataFrame = data
    self.config: ModelConfig = config
    self.subset: list[str] = subset
    self.subset_name: str = subset_name
    ## optimizer setup ##
    self.features: list[str] = []
    self.bgs: list[float] = []
    self.bounds: list[Tuple[float, float]] = []
    self.tol: float = tol
    self.step: float = step
    self.method: str = method
    self.init_features()
    ## in-optimization data ##
    self.round_number: int = 0
    self.optimization_records: list[dict] = []
    self.best_mae: Optional[float] = None
    ## post optimization data ##
    self.solution: Any = None
    self.optimization_results: dict = {}
  
  def normalize_param(self, value: float, param: ModelParam) -> float:
    '''
    Normalize a parameter to a value between 0 and 1.
    '''
    return (value - param.opti_min) / (param.opti_max - param.opti_min)
  
  def denormalize_param(self, value: float, param: ModelParam) -> float:
    '''
    Denormalize a parameter from a value between 0 and 1.
    '''
    return value * (param.opti_max - param.opti_min) + param.opti_min
  
  def denormalize_optimizer_values(self, x: list[float]) -> dict:
    '''
    Denormalizes an optimizer values list into a config dictionary.
    '''
    ## start with a copy of the config ##
    local_config = self.config.values.copy()
    ## update the config with the new values that are being optimized ##
    for i, k in enumerate(x):
      local_config[self.features[i]] = self.denormalize_param(k, self.config.params[self.features[i]])
    return local_config
  
  def init_features(self):
    '''
    Initialize the features, bgs, and bounds for the optimizer.
    Allows for a subset of features to be optimized if a subset is provided.
    '''
    for k, v in self.config.params.items():
      if len(self.subset) > 0 and k not in self.subset:
        continue
      self.features.append(k)
      self.bgs.append(self.normalize_param(v.value, v))
      self.bounds.append((0,1)) ## all features are normalized ##

  def objective(self, x: list[float]) -> float:
    '''
    Objective function for the optimizer.
    '''
    ## increment the round number ##
    self.round_number += 1
    ## initialize a QBModel ##
    model = QBModel(self.data, self.denormalize_optimizer_values(x))
    ## run the model ##
    model.run_model()
    ## score the model ##
    scored_record = model.score_model(add_elo=False)
    ## add the record to the optimization records ##
    self.optimization_records.append(scored_record)
    ## save the record if it is a new best, or if it an interval of 100 rounds ##
    save_record = False
    if self.best_mae is None:
      self.best_mae = scored_record['mae']
      save_record = True
    elif scored_record['mae'] < self.best_mae:
      self.best_mae = scored_record['mae']
      save_record = True
    if self.round_number % 100 == 0:
      save_record = True
    if save_record:
      df = pd.DataFrame(self.optimization_records)
      df.to_csv('{0}/In-flight Results/{1}{2}.csv'.format(
        pathlib.Path(__file__).parent.resolve(),
        datetime.datetime.now().strftime('%Y%m%d'),
        '_{0}'.format(self.subset_name) if len(self.subset) > 0 else ''
      ))
    ## calculate the objective ##
    return scored_record['mae']

  def update_config(self, x: list[float]):
    '''
    Update the config with the new values and save the result.
    '''
    ## create the updated config ##
    updated_config = self.denormalize_optimizer_values(x)
    ## apply additional rounding ##
    for k, v in updated_config.items():
      updated_config[k] = round(v, 6)
    ## update the config ##
    self.config.update_config(updated_config)
    ## save to package config file ##
    self.config.to_file()
    ## save the result ##
    self.save_result()
  
  def optimize(
      self,
      save_result: bool = True,
      update_config: bool = False
    ):
    '''
    Core optimization function.
    '''
    ## run the optimizer ##
    ## start timer ##
    start_time = float(time.time())
    solution = minimize(
      self.objective,
      self.bgs,
      bounds=self.bounds,
      method=self.method,
      options={
          'ftol' : self.tol,
          'eps' : self.step
      }
    )
    ## end timer ##
    end_time = float(time.time())
    ## save the solution ##
    self.solution = solution
    ## create an optimization result object ##
    ## values ##
    optimal_config = self.denormalize_optimizer_values(solution.x)
    ## add objective function reached ##
    self.optimization_results['mae'] = solution.fun
    self.optimization_results['runtime'] = end_time - start_time
    ## extend the optimization results with the optimal config ##
    self.optimization_results = self.optimization_results | optimal_config
    ## save as needed ##
    if save_result:
      df = pd.DataFrame([self.optimization_results])
      df.to_csv('{0}/Results/{1}{2}.csv'.format( 
        pathlib.Path(__file__).parent.resolve(),
        datetime.datetime.now().strftime('%Y%m%d'),
        '_{0}'.format(self.subset_name) if len(self.subset) > 0 else ''
      ))
    ## update the config if needed ##
    if update_config:
      self.update_config(solution.x)