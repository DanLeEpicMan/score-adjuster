'''
This module contains several tools for adjusting scores. 
All methods assume sufficient preprocessing has occured.
See `score_tools.py` for preprocessing tools.

Meant for Data Science UCSB's events. Written by Daniel Naylor.

# Requirements
    - Numpy   >= 2.2.5    (https://pypi.org/project/numpy/)
    - Pandas  >= 2.2.3    (https://pypi.org/project/pandas/)
    - GSheets >= 0.6.1    (https://pypi.org/project/gsheets/)
        - Only required if used in conjunction with `score_tools.py`


# Methods
    - `kappa_adjusted`:              The Kappa-Adjusted score. This is intended to up-weigh judges who
                                     utilize the full spectrum of scores. Kappa is a hyperparameter.

    - `proportional_variance`:       The Proportional-Variance score. This has the same intentions
                                     as Kappa-Adjusted, albeit mathematically different.

    - `showcase_prescreening_score`: A weighted average between attendance and adjusted score.
'''
import pandas as pd
import numpy as np


def _normalize(arr: np.ndarray | pd.Series, /) -> np.ndarray | pd.Series:
    '''
    f: [a,b] -> [0, 1]
    f(x) = (x - a) / (b - a)
    '''
    a = arr.min()
    b = arr.max()

    return (arr - a) / (b - a)


def kappa_adjusted(
        scores: pd.DataFrame,
        *,
        kappa: int = 1,
        normalize: bool = True
) -> pd.Series:
    '''
    Compute the Kappa-Adjusted score of a given dataset of scores.
    See associated commentary for a definition and other explanations.

    ## Parameters
        - `scores`:       A DataFrame of project scores. Each column is assumed to be a judge, while
                        each row is assumed to be a single project. The row labels should be the project names.
        - `kappa`:        A hyperparameter for the score adjustment. 
                        Note that the ordering of `kappa=0` is equivalent to zero adjustment, i.e. using raw scores.
        - `normalize`:    Whether to normalize the final scores onto a range of `[0, 1]`. Default is True.
    ## Return
        A pandas series object, whose indices are `scores` indices, and whose values are the adjusted score.
        Note that this series is sorted in descending order.
    '''
    avg_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    adjusted_scores = np.sum(
        np.sign(scores - avg_scores) * (np.abs(scores - avg_scores) * std_scores**kappa) ** (1 / (1 + kappa)),
        axis=1
    )
    normalized_scores = _normalize(adjusted_scores) if normalize else adjusted_scores

    return normalized_scores.sort_values(ascending=False)

def proportional_variance(
        scores: pd.DataFrame,
        *,
        normalize: bool = True
    ) -> pd.Series:
    '''
    Compute the Proportional-Variance score of a given dataset of scores.
    This is essentially a weighted average of judges, where the weights are the proportional variances.
    See associated commentary for a definition and other explanations.

    ## Parameters
        - `scores`:       A DataFrame of project scores. Each column is assumed to be a judge, while
                        each row is assumed to be a single project. The row labels should be the project names.
        - `normalize`:    Whether to normalize the final scores onto a range of `[0, 1]`. Default is True.
    ## Return
        A pandas series object, whose indices are `scores` indices, and whose values are the adjusted score.
        Note that this series is sorted in descending order.
    '''
    std_scores = scores.var(axis=0)
    weights = std_scores / std_scores.sum()

    adjusted_scores = np.sum(
        weights * scores, # multiply each judge's score by their weights
        axis=1
    )
    normalized_scores = _normalize(adjusted_scores) if normalize else adjusted_scores

    return normalized_scores.sort_values(ascending=False)

def showcase_prescreening_score(
        scores: pd.Series,
        attendance: pd.Series,
        *,
        attendance_ratio: float = 0.2,
        mapping: dict[str, int] | None = None
) -> pd.Series:
    '''
    Compute an adjusted score weighted between `scores` and `attendance`.
    This is called "showcase_prescreening_score" because this is the score used for
    prescreening projects in project showcase. I could not think of a better name.

    ## Parameters
        - `scores`: Adjusted scores. These should be on a range of [0, 1] for this method to work properly.
        See `kappa_adjusted` or `proportional_variance`.
        - `attendance`: Attendance scores. Expected to be on a range of [0, 1].
        See `get_attendance_scores`.
        - `attendance_ratio`: The percentage that attendance will contribute to the final score. 
        - `mapping`: A mapping from the indicies of `score` to the indices of `attendance`. Typically project name to number.
        If given `None`, then this assumes that the indices are the same. See `get_name_number_pairs`.
    '''
    if mapping is not None:
        attendance = pd.Series(attendance.values, index=[mapping[x] for x in attendance.index])

    return scores * (1 - attendance_ratio) + attendance.loc[scores.index] * attendance_ratio

if __name__ == '__main__':
    import score_tools

    test_scores = score_tools.get_project_scores('data.csv')

    adjusted = proportional_variance(test_scores)
    
    print(adjusted)