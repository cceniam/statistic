import pandas as pd
import numpy as np

class Person:
    age = 0
    gender = 0  # 性别： 0 男， 1 女
    age_of_onset = 0    # 初次发病年纪
    duration = 0    # 病程
    pre_hospitalization = 0 # 既往住院次数
    education = 0   # 0:高中及以下 , 1 大专及以上
    pnass = pd.DataFrame(

    )
    class PANSS:
        ps = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
        ns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
        gs = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15','G16']
        def calculate_subscale_scores(self,df):
            """
            计算阳性、阴性和一般精神病理学的评分
            :param df: 包含PANSS评分的DataFrame
            :return: 新的DataFrame，包含子量表分数
            """
            df['positive_score'] = df[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].sum(axis=1)
            df['negative_score'] = df[['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']].sum(axis=1)
            df['general_score'] = df[
                ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15',
                 'G16']].sum(axis=1)
            df['total_score'] = df['positive_score'] + df['negative_score'] + df['general_score']
            return df