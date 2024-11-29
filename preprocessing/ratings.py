import numpy as np
import pandas as pd


def company_popularity(data):  # Популярность компании
    return data.groupby('client_name').size()


def staff_turnover(data):  # Текучка кадров
    return data[data.grade_proof == 'подтверждён'].groupby('client_name').size()


def competition_ratio(data):
    competition = data.groupby(['client_name', 'position']).size()
    confirmed_count = data[data['grade_proof'] == 'подтверждён'].groupby(['client_name', 'position']).size()
    competition_df = competition.to_frame(name='total_applicants').join(
        confirmed_count.to_frame(name='confirmed_applicants'), how='left'
    )
    competition_df['confirmed_applicants'] = competition_df['confirmed_applicants'].fillna(0)  # Заполнить NaN нулями
    competition_df['competition_ratio'] = competition_df['total_applicants'] / competition_df['confirmed_applicants']
    competition_df['competition_ratio'].replace(np.inf, np.nan,
                                                inplace=True)  # Убрать бесконечности (если нет подтверждённых)

    # competition_df[competition_df.competition_ratio != np.nan]
    return competition_df


def calculate_complany_rating(data):
    pass
