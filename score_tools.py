'''
This module contains several tools for preprocessing scores. 
The purpose of this module is to process scores into a format that
methods in `score_adjustments.py` can accept.

Meant for Data Science UCSB's events. Written by Daniel Naylor.

# Requirements
  - Pandas  >= 2.2.3    (https://pypi.org/project/pandas/)
  - GSheets >= 0.6.1    (https://pypi.org/project/gsheets/)

# Methods
    - `get_project_scores`:     Return a DataFrame of raw scores that can be passed into other methods
    - `get_attendance_scores`:  Return a Series of attendance scores.
    - `get_name_number_pairs`:  Return a dictionary mapping Project Names to Project Numbers.
'''
import pandas as pd
import gsheets as gs


def get_project_scores(
        csv_or_frame: str | pd.DataFrame | None = None,
        *,
        sheet_id: str | None = None,
        credentials: str | None = None,
        index: str | list = 'Project Name',
        ignored_cols: list[str] = ['Project Number']
    ) -> pd.DataFrame:
    '''
    Given a spreadsheet ID, CSV file, or DataFrame,
    return a DataFrame of raw scores, suitable for methods in `project_selector.py`.

    ## Parameters
    - `sheet_id`:       The ID of the Google sheet. The first page will be read.
                        Mutually exclusive with `csv_or_frame`.

    - `credentials`:    JSON file for Google API credentials. 
                        Required if `sheet_id` is given.

    - `csv_or_frame`:   A path to a CSV file or a pandas DataFrame. 
                        Mutually exclusive with `sheet_id`.

    - `index`:          The index column for the DataFrame. This may be a sufficiently sized list.
                        This will be the index for the returned frame.

    - `ignored_cols`:   By default, this works by dropping every non-numeric column, 
                        and treating numeric ones as a judge's scores. 
                        To avoid including numeric columns unrelated to scores, 
                        use the `ignored_cols` parameter.
    
    ##  Returns
        
    A Pandas DataFrame, whose row indices is `proj_names`.

    ## Raises
    - `ValueError`: Both `sheet_id` and `csv_or_frame` are given.
    - `TypeError`: `sheet_id` is given but not `credentials` (or vice versa).
    '''
    if sheet_id is not None and csv_or_frame is not None:
        raise ValueError('Both "sheet_id" and "csv_or_frame" are given.')
    
    if (sheet_id is not None and credentials is None) or (sheet_id is None and credentials is not None):
        raise TypeError('Missing one of "sheet_id" or "credentials"')
    
    if sheet_id is not None:
        s = gs.Sheets.from_files(credentials)
        raw_data: pd.DataFrame = s.get(sheet_id).first_sheet.to_frame()
    else:
        raw_data = pd.read_csv(csv_or_frame) if type(csv_or_frame) == str else csv_or_frame.copy()

    raw_data.set_index(
        index, 
        inplace=True,
        drop=True
    )
    raw_data.drop(ignored_cols, inplace=True, axis=1)
    return raw_data.select_dtypes(['number'])

def get_attendance_scores(
        csv_or_frame: str | pd.DataFrame | None = None,
        *,
        sheet_id: str | None = None,
        credentials: str | None = None,
        page_num: int = 0,
        index: str | list = 'Project Number',
        ignored_cols: list[str] = []
) -> pd.Series:
    '''
    Given a spreadsheet ID, CSV file, or DataFrame,
    compute the attendance score (Present / Total Checkins).

    The input is expected to be a spreadsheet with the following format:
        - One index column, whose name is given in the parameter `index`. This may be absent.
        - Each column is a date, each row a project group
        - Every absent is marked with NaN (empty)
        - Every present is filled (doesn't matter with what)

    It is recommended that the index be the project numbers.
    ## Parameters
    - `sheet_id`:       The ID of the Google sheet. The first page will be read.
                        Mutually exclusive with `csv_or_frame`.

    - `credentials`:    JSON file for Google API credentials. 
                        Required if `sheet_id` is given.

    - `page_num`:       The zero-indexed page number for attendance.
                        Only used if `sheet_id` is given, otherwise ignored.

    - `csv_or_frame`:   A path to a CSV file or a pandas DataFrame. 
                        Mutually exclusive with `sheet_id`.

    - `index`:          The index column. A list of indices may be passed instead.

    - `ignored_cols`:   Columns to ignore during computation. This is meant to
                        allow the presence of additional columns without requiring their removal
                        in the base spreadsheet.
    
    ##  Returns
        
    A Pandas Series, whose entires are the attendance scores (Presents / Total Checkins).
    Note that this series is **not** sorted.

    ## Raises
    - `ValueError`: Both `sheet_id` and `csv_or_frame` are given.
    - `TypeError`: `sheet_id` is given but not `credentials` (or vice versa).
    '''
    if sheet_id is not None and csv_or_frame is not None:
        raise ValueError('Both "sheet_id" and "csv_or_frame" are given.')
    
    if (sheet_id is not None and credentials is None) or (sheet_id is None and credentials is not None):
        raise TypeError('Missing one of "sheet_id" or "credentials"')
    
    if sheet_id is not None:
        s = gs.Sheets.from_files(credentials)
        raw_data: pd.DataFrame = s.get(sheet_id).sheets[page_num].to_frame()
    else:
        raw_data = pd.read_csv(csv_or_frame) if type(csv_or_frame) == str else csv_or_frame.copy()

    raw_data.set_index(
        index, 
        inplace=True,
        drop=True
    )
    raw_data.drop(ignored_cols, inplace=True, axis=1)

    return raw_data.count(axis=1) / raw_data.shape[1]

def get_number_name_pairs(
        csv_or_frame: str | pd.DataFrame | None = None,
        *,
        sheet_id: str | None = None,
        credentials: str | None = None,
        name_col: str = 'Project Name',
        num_col: str = 'Project Number'
) -> dict[int, str]:
    '''
    Returns a dictionary mapping a project's number to a project name.

    ## Parameters
    - `sheet_id`:       The ID of the Google sheet. The first page will be read.
                        Mutually exclusive with `csv_or_frame`.

    - `credentials`:    JSON file for Google API credentials. 
                        Required if `sheet_id` is given.

    - `csv_or_frame`:   A path to a CSV file or a pandas DataFrame. 
                        Mutually exclusive with `sheet_id`.

    - `name_col`:       The column containing the project names.

    - `num_col`:        The column containing the project numbers.
    '''
    if sheet_id is not None and csv_or_frame is not None:
        raise ValueError('Both "sheet_id" and "csv_or_frame" are given.')
    
    if (sheet_id is not None and credentials is None) or (sheet_id is None and credentials is not None):
        raise TypeError('Missing one of "sheet_id" or "credentials"')
    
    if sheet_id is not None:
        s = gs.Sheets.from_files(credentials)
        raw_data: pd.DataFrame = s.get(sheet_id).first_sheet.to_frame()
    else:
        raw_data = pd.read_csv(csv_or_frame) if type(csv_or_frame) == str else csv_or_frame

    return dict(zip(raw_data[num_col], raw_data[name_col]))

if __name__ == '__main__':
    test = pd.read_csv('data.csv')

    print(test, get_project_scores(csv_or_frame=test), sep='\n\n')