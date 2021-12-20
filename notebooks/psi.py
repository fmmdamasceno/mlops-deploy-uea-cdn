import numpy as np
import pandas as pd

def create_psi_table(actual, expected):
    
    psi_df = pd.DataFrame({
        'classes': list(actual.unique()),
        'bucket': list(range(1,len(list(actual.unique()))+1)),
        'actual_count': list(actual.value_counts().sort_index().values),
        'expected_count': list(expected.value_counts().sort_index().values),
        'actual_percent': list(actual.value_counts(normalize=True).sort_index().values),
        'expected_percent': list(expected.value_counts(normalize=True).sort_index().values)
    })

    psi_df['actual-expected'] = psi_df['actual_percent']-psi_df['expected_percent']
    psi_df['ln(actual/percent)'] = np.log(psi_df['actual_percent']/psi_df['expected_percent'])
    psi_df['psi'] = psi_df['actual-expected']*psi_df['ln(actual/percent)']
        
    return psi_df


def psi(actual, expected):
    return round(sum(create_psi_table(actual, expected)['psi']), 5)

def psi_interpretation(actual, expected):
    df_interpretation = pd.DataFrame({
        'scales': ['PSI < 0.1', 'PSI < 0.2', 'PSI >= 0.2'],
        'description': ['no significant population change', 'moderate population change', 'significant population change'],
    })
    
    psi_value = psi(actual, expected)
    
    if psi_value < 0.1:
        you_scale = ['X', '', '']
    elif psi_value < 0.2:
        you_scale = ['', 'X', '']
    else:
        you_scale = ['', '', 'X']
    
    df_interpretation['your_scale'] = you_scale
    
    return df_interpretation