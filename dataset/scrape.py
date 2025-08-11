from typing import Optional, Union
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import requests
from enum import Enum
from pathlib import Path
from tqdm.auto import tqdm
from itertools import product
import warnings
import csv

warnings.filterwarnings('ignore')


class StatTypes(Enum):
  '''
    Stat IDs from api.collegefootballdata.com/plays/stats/types
    '''
  INCOMPLETION = 1
  TARGET = 2
  PASS_BREAKUP = 3
  COMPLETION = 4
  RECEPTION = 5
  TACKLE = 6
  RUSH = 7
  FUMBLE = 8
  FUMBLE_FORCED = 9
  FUMBLE_RECOVERED = 10
  SACK_TAKEN = 11
  SACK = 12
  KICKOFF = 13
  ONSIDE_KICK = 14
  KICKOFF_RETURN = 15
  PUNT = 16
  PUNT_BLOCK = 17
  FG_ATTEMPT_BLOCKED = 18
  FIELD_GOAL_BLOCK = 19
  INTERCEPTION_THROWN = 20
  INTERCEPTION = 21
  TOUCHDOWN = 22
  FIELD_GOAL_ATTEMPT = 23
  FIELD_GOAL_MADE = 24
  FIELD_GOAL_MISSED = 25


class CFBDClient:
  '''
    Creates a simple client to interact with the College Football Data API.
  '''
  hostname: str
  headers: dict[str, str]

  def __init__(self, hostname: str, bearer_token: str):
    self.hostname = hostname
    self.headers = dict()
    self.headers['Authorization'] = f'Bearer {bearer_token}'

  def get(
    self,
    subpath: str,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, any]] = None,
    as_df: bool = False,
  ) -> Union[dict, pd.DataFrame]:
    if headers is None:
      headers = {}
    headers.update(self.headers)

    response = requests.get(
      url=f'{self.hostname}/{subpath}',
      headers=headers,
      params=params,  # <-- Pass query parameters here
    )
    if not response.ok:
      raise requests.exceptions.RequestException(
        f'{response.status_code}: {response.text}', response=response)
    json_data = response.json()
    return json_data if not as_df else self.response_as_df(json_data)

  @staticmethod
  def response_as_df(response: Union[dict, list[dict]]) -> pd.DataFrame:
    '''
        Convert API JSON response (dict or list of dicts) into a Pandas DataFrame.
        '''
    if isinstance(response, dict):
      # If dict has a single top-level key pointing to list of dicts
      # or just a flat dict
      if (all(isinstance(v, list) for v in response.values()) and
          len(response) == 1):
        response = next(iter(response.values()))
      else:
        response = [response]
    return pd.DataFrame(response)


def get_field_goal_data(year: int,
                        week: int,
                        export_folder_path: Path,
                        retrieve_weather_data=True) -> None:
  '''
    Get field goal data for given `year` and `week`
    '''
  # Export settings
  assert export_folder_path.is_dir(
  ), 'Error: export_folder_path must be a folder.'
  export_name = f'fg_data_yr{year}_wk{week}_wt{1 if retrieve_weather_data else 0}.csv'
  export_path = str(export_folder_path / export_name)

  # Retrieve makes
  makes = client.get(
    'plays/stats',
    params={
      'year': year,
      'week': week,
      'statTypeId': StatTypes.FIELD_GOAL_MADE.value,
      'classification': 'fbs',
      'seasonType': 'regular',
    },
    as_df=True,
  )

  # Retrieve attempts
  attempts = client.get(
    'plays/stats',
    params={
      'year': year,
      'week': week,
      'statTypeId': StatTypes.FIELD_GOAL_ATTEMPT.value,
      'classification': 'fbs',
      'seasonType': 'regular',
    },
    as_df=True,
  )

  # Merge field goal data
  field_goals = pd.merge(
    left=attempts,
    right=makes[['playId', 'statType']],
    how='left',
    on='playId',
  )
  field_goals.rename(columns={
    'statType_x': 'statType',
    'statType_y': 'made'
  },
                     inplace=True)

  # Convert made column to 0/1 indicator
  field_goals['made'] = field_goals['made'].fillna(0)
  field_goals['made'] = field_goals['made'].replace({
    'Field Goal Made': 1,
    0: 0
  })
  field_goals['made'] = field_goals['made'].astype(int)

  if not retrieve_weather_data:
    field_goals.to_csv(export_path, index=False)
    return

  # Retrieve weather data
  weather_columns_to_keep = [
    'id',
    'gameIndoors',
    'homeTeam',
    'awayTeam',
    'temperature',
    'dewPoint',
    'humidity',
    'precipitation',
    'snowfall',
    'windDirection',
    'windSpeed',
    'pressure',
    'weatherConditionCode',
    'weatherCondition',
  ]

  weather = client.get(
    'games/weather',
    params={
      'year': year,
      'week': week,
      'classification': 'fbs',
      'seasonType': 'regular',
    },
    as_df=True,
  )

  # Merge weather data
  # NOTE: this merge only keeps samples that also include weather
  #       data: samples without weather data are dropped.
  merged = pd.merge(
    left=field_goals,
    right=weather[weather_columns_to_keep],
    how='inner',
    left_on='gameId',
    right_on='id',
  )
  merged.drop(columns='id', inplace=True)

  merged.to_csv(export_path, index=False)


def get_field_goal_data_between(year_span: range, week_span: range,
                                export_folder_path: Path):
  '''
  Get field goal data for given year and week span.
  '''
  total = len(year_span) * len(week_span)
  with tqdm(total=total) as pbar:
    for year, week in product(year_span, week_span):
      pbar.set_description(f'Exporting {year} wk. {week}')
      get_field_goal_data(year, week, export_folder_path)
      pbar.update(1)
  print(f'EXPORT FINISHED: WROTE {total} FILES')


def merge_csvs(data_folder: Path, export_path: Path) -> None:
  '''
  Merges all csv files at `data_folder` and writes a new csv file to `export_path`.
  '''
  csv_files = sorted(data_folder.glob('*.csv'))
  if not csv_files:
    raise ValueError(f'No CSV files found in {data_folder}')

  header = None
  merged_rows = []

  for i, file in enumerate(csv_files):
    with file.open(newline='', encoding='utf-8') as f:
      reader = csv.reader(f)
      file_header = next(reader)

      if i == 0:
        header = file_header
        merged_rows.extend(reader)
      else:
        if file_header != header:
          raise ValueError(f'Header mismatch in {file.name}')
        merged_rows.extend(reader)

  with export_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(merged_rows)
  print(f'{data_folder}: Merged --> {export_path}')


def purge_folder(folder: Path) -> None:
  '''
    Recursively deletes all files and subdirectories inside `folder`.
    '''
  if not folder.exists():
    return

  for path in folder.rglob('*'):
    if path.is_file() or path.is_symlink():
      path.unlink()
    elif path.is_dir():
      purge_folder(path)  # Recursively delete contents
      path.rmdir()
  folder.rmdir()


def validate_dir(folder: Path) -> None:
  '''
    Validate that the directory at `folder` exists. Creates
    one if it does not exist.
    '''
  if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)
  elif not folder.is_dir():
    raise NotADirectoryError(f'{folder} exists but is not a directory')


if __name__ == '__main__':
  load_dotenv()

  client = CFBDClient(
    hostname='https://api.collegefootballdata.com/',
    bearer_token=os.getenv('CFBDATA_API_KEY'),
  )

  data_folder = Path('dataset/data')
  validate_dir(data_folder)

  get_field_goal_data_between(
    year_span=range(2021, 2025),  # post-COVID
    week_span=range(0, 16),  # regular season
    export_folder_path=data_folder,
  )

  merge_csvs(data_folder, Path('dataset/fg_data.csv'))

  purge_folder(data_folder)
