import random

import pandas as pd
import numpy as np


class Person:
    age = 0
    gender = 0  # 性别： 0 男， 1 女
    age_of_onset = 0  # 初次发病年纪
    duration = 0  # 病程
    pre_hospitalization = 0  # 既往住院次数
    education = 0  # 0:高中及以下 , 1 大专及以上
    pnass = pd.DataFrame(

    )

    class PANSS:
        ps = [['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'], []]
        ns = [['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'], []]
        gs = [['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16'],
              []]

        def calculate_subscale_scores(self, df):
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

        def generate_test_data(self, positive_score, negative_score, general_score, total_score):
            """
            根据指定的positive_score, negative_score, general_score, total_score生成测试数据
            :param positive_score: 指定的阳性评分
            :param negative_score: 指定的阴性评分
            :param general_score: 指定的一般精神病理评分
            :param total_score: 指定的总评分
            :return: 一个DataFrame，包含生成的测试数据
            """
            # 定义子评分的数量
            ps_count = len(self.ps[0])
            ns_count = len(self.ns[0])
            gs_count = len(self.gs[0])

            # 定义合理的评分范围
            min_score, max_score = 1, 7

            # 随机生成满足要求的各项分数，并且每项分数在合理范围内
            def random_partition(target_sum, parts, min_value=min_score, max_value=max_score):
                while True:
                    values = [random.randint(min_value, max_value) for _ in range(parts)]
                    if sum(values) == target_sum:
                        return values

            df = pd.DataFrame(columns=self.ps[0] + self.ns[0] + self.gs[0])

            while True:
                positive_values = random_partition(positive_score, ps_count)
                negative_values = random_partition(negative_score, ns_count)
                general_values = random_partition(general_score, gs_count)

                # 将分数组合成DataFrame的行数据
                data = {**dict(zip(self.ps[0], positive_values)),
                        **dict(zip(self.ns[0], negative_values)),
                        **dict(zip(self.gs[0], general_values))}

                df = df.append(data, ignore_index=True)
                df = self.calculate_subscale_scores(df)

                if df['total_score'].iloc[-1] == total_score:
                    break

            return df.iloc[[-1]]
