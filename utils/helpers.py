import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import neattext as nt
import neattext.functions as nfx
import fuzzywuzzy
from fuzzywuzzy import process
import charset_normalizer
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# EXPLORATORY DATA ANALYSIS

# check dataframe

def check_df(dataframe, head=10):
    print('#' * 20, 'Head', '#' * 20)
    print(dataframe.head(head))
    print('#' * 20, 'Shape', '#' * 20)
    print(dataframe.shape)
    print('#' * 20, 'Data Info', '#' * 20)
    print(dataframe.info())
    print('#' * 20, 'Data Types', '#' * 20)
    print(dataframe.dtypes)
    print('#' * 20, 'Missing Values', '#' * 20)
    print(dataframe.isnull().sum().sort_values(ascending=False))
    print('#' * 20, 'Descriptive Statistics', '#' * 20)
    print(dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

# categorical, numerical and cardinal variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                    dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                    dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

# summary of categorical variables
def cat_summary(dataframe, col_name, plot=False):
    print(f'\n{col_name.capitalize()} Summary:')
    counts = dataframe[col_name].value_counts()
    percentages = counts / len(dataframe) * 100
    print(pd.concat([counts, percentages], axis=1, keys=['Count', 'Percentage']))

    if plot:
        dataframe[col_name].value_counts().plot(kind='bar', rot=90)
        plt.xlabel(col_name.capitalize())
        plt.ylabel('COUNT')
        plt.show()

# summary of numerical variables
def num_summary(dataframe, numerical_col, plot=False):
    print(f'\n{numerical_col.capitalize()} Summary:')
    print(dataframe[numerical_col].describe().T.round(2))

    if plot:
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=dataframe, y=numerical_col)
        plt.xlabel(numerical_col.capitalize())
        plt.subplot(1, 2, 2)
        sns.histplot(data=dataframe, x=numerical_col)
        plt.xlabel(numerical_col.capitalize())
        plt.show()

# Analysis of target variable with categorical variables
def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print('\n', '#' * 10, categorical_col.capitalize(), '#' * 10)
    target_mean = round(dataframe.groupby(categorical_col)[target].mean(), 2)
    print(pd.DataFrame({'TARGET_MEAN': target_mean}))

    if plot:
        sns.barplot(x=dataframe[categorical_col], y=dataframe[target], ci=False)
        plt.xlabel(categorical_col.capitalize())
        plt.ylabel(target.capitalize())
        plt.show()

# Analysis of target variable with numerical variables
def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    # target_mean = dataframe.groupby(target)[numerical_col].mean()
    # print(target_mean)

    if plot:
        sns.scatterplot(x=dataframe[numerical_col], y=dataframe[target])
        plt.xlabel(numerical_col.capitalize())
        plt.ylabel(target.capitalize())
        plt.show()

# high_correlated_cols' function to delete the variables with high correlation
def high_correlated_cols(dataframe, corr_threshold=0.90, plot=False):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_threshold)]

    if plot:
        plt.figure(figsize=(20, 15))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, cmap='RdBu', annot=True, linewidths=0.5, linecolor='w', mask=mask)
        plt.show()

    return drop_list

# DATA PREPROCESSING

# finding outlier thresholds
def outlier_thresholds(dataframe, col_name, lower_quantile=0.05, upper_quantile=0.95):
    lower_quartile = dataframe[col_name].quantile(lower_quantile)
    upper_quartile = dataframe[col_name].quantile(upper_quantile)
    iqr = upper_quartile - lower_quartile
    upper_limit = round(upper_quartile + 1.5 * iqr, 2)
    lower_limit = round(lower_quartile - 1.5 * iqr, 2)
    return lower_limit, upper_limit

# replace outliers with thresholds
def replace_with_thresholds(dataframe, variable, lower_quantile=0.05, upper_quantile=0.95):
    lower_quartile = dataframe[variable].quantile(lower_quantile)
    upper_quartile = dataframe[variable].quantile(upper_quantile)
    iqr = upper_quartile - lower_quartile
    upper_limit = round(upper_quartile + 1.5 * iqr, 2)
    lower_limit = round(lower_quartile - 1.5 * iqr, 2)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit

# is there any outliers?
def has_outliers(dataframe, col_name, lower_quantile=0.05, upper_quantile=0.95):
    lower_quartile = dataframe[col_name].quantile(lower_quantile)
    upper_quartile = dataframe[col_name].quantile(upper_quantile)
    iqr = upper_quartile - lower_quartile
    upper_limit = round(upper_quartile + 1.5 * iqr, 2)
    lower_limit = round(lower_quartile - 1.5 * iqr, 2)
    return dataframe[
        ((dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit))].any(axis=None)

# printing outliers
def print_outliers(dataframe, column_name, lower_quantile=0.05, upper_quantile=0.95, show_index=False):
    lower_quartile = dataframe[column_name].quantile(lower_quantile)
    upper_quartile = dataframe[column_name].quantile(upper_quantile)
    iqr = upper_quartile - lower_quartile
    upper_limit = round(upper_quartile + 1.5 * iqr, 2)
    lower_limit = round(lower_quartile - 1.5 * iqr, 2)
    if dataframe[
        ((dataframe[column_name] < lower_limit) | (dataframe[column_name] > upper_limit))].shape[0] > 10:
        print(dataframe[((dataframe[column_name] < lower_limit) | (
                    dataframe[column_name] > upper_limit))].head())
    else:
        print(dataframe[
                    ((dataframe[column_name] < lower_limit) | (dataframe[column_name] > upper_limit))])

    if show_index:
        outlier_index = dataframe[
            ((dataframe[column_name] < lower_limit) | (dataframe[column_name] > upper_limit))].index
        return outlier_index

# removing outliers
def remove_outliers(dataframe, column_name, lower_quantile=0.05, upper_quantile=0.95):
    lower_quartile = dataframe[column_name].quantile(lower_quantile)
    upper_quartile = dataframe[column_name].quantile(upper_quantile)
    iqr = upper_quartile - lower_quartile
    upper_limit = round(upper_quartile + 1.5 * iqr, 2)
    lower_limit = round(lower_quartile - 1.5 * iqr, 2)
    df_without_outliers = dataframe[
        ~((dataframe[column_name] < lower_limit) | (dataframe[column_name] > upper_limit))]
    return df_without_outliers

# missing values
def missing_values_table(dataframe, return_cols=False):
    missing_cols = dataframe.columns[dataframe.isnull().any()]
    missing_count = dataframe[missing_cols].isnull().sum().sort_values(ascending=False)
    missing_ratio = (missing_count / dataframe.shape[0]) * 100
    missing_data = pd.concat([missing_count, missing_ratio], axis=1, keys=['Missing Count', 'Missing Ratio (%)'])
    print(missing_data, end="\n")
    if return_cols:
        return missing_cols

# missing values vs target
def missing_vs_target(dataframe, target, missing_cols):
    temp_df = dataframe.copy()
    for col in missing_cols:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.columns[temp_df.columns.str.contains("_NA_FLAG")]
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

# label encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# one-hot encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# rare analyzer
def rare_analyzer(dataframe, target, categorical_columns, threshold=0.05):
    for col in categorical_columns:
        counts = dataframe[col].value_counts(normalize=True)
        rare_labels = counts[counts < threshold].index
        print(f'{col} : {len(rare_labels)}')
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(),
                            'RATIO': dataframe[col].value_counts(normalize=True),
                            'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')

# encoding rare colums
def rare_encoder(dataframe, rare_perc=0.05):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if
                    temp_df[col].dtypes == 'O' and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(
                        axis=None)]
    for col in rare_columns:
        counts = temp_df[col].value_counts() / len(temp_df)
        rare_labels = counts[counts < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])
    return temp_df

def fuzzy_closest_matches(dataframe, word, column_name, limit):
    # get the top 10 closest matches to given word
    # convert to lower case
    temp_df = dataframe.copy()
    temp_df[column_name] = temp_df[column_name].str.lower()
    # remove trailing white spaces
    temp_df[column_name] = temp_df[column_name].str.strip()

    closest_matches = fuzzywuzzy.process.extract(word, temp_df[column_name].unique(), limit=limit,
                                                    scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    return closest_matches

def replace_matches_in_column(dataframe, column, string_to_match, limit=10, min_ratio=45):
    # get a list of unique strings
    temp_df = dataframe.copy()

    strings = temp_df[column].unique()

    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                            limit=limit, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = temp_df[column].isin(close_matches)

    # replace all rows with close matches with the input matches
    temp_df.loc[rows_with_matches, column] = string_to_match

    # let us know the function's done
    print("All done!")
    return temp_df

def neattext_nlp(dataframe, column, puncts=True, stopwords=True, urls=True, emails=True,
                    numbers=False, multiple_whitespaces=False, currency_symbols=True,
                    special_char=True):
    temp_df = dataframe.copy()
    # Noise Scan # Scan Percentage of Noise(Unclean data) in text
    print('#' * 20, "\n")
    print("Before preprocess: ", temp_df[column].apply(lambda x: nt.TextFrame(x).noise_scan()["text_noise"]))
    temp_df[column] = temp_df[column].apply(
        lambda x: nfx.clean_text(x, puncts=puncts, stopwords=stopwords, urls=urls, emails=emails,
                                    numbers=numbers, multiple_whitespaces=multiple_whitespaces,
                                    currency_symbols=currency_symbols, special_char=special_char))
    print('#' * 20, "\n")
    print("After preprocess: ", temp_df[column].apply(lambda x: nt.TextFrame(x).noise_scan()["text_noise"]))
    # remove trailing white spaces
    temp_df[column] = temp_df[column].str.strip()
    return temp_df


