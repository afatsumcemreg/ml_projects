import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

class preprocessing():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    # EXPLORATORY DATA ANALYSIS

    # check dataframe
    
    def check_df(self, head=5):
        print('#' * 20, 'Head', '#' * 20)
        print(self.dataframe.head(head))
        print('#' * 20, 'Shape', '#' * 20)
        print(self.dataframe.shape)
        print('#' * 20, 'Data Info', '#' * 20)
        print(self.dataframe.info())
        print('#' * 20, 'Data Types', '#' * 20)
        print(self.dataframe.dtypes)
        print('#' * 20, 'Missing Values', '#' * 20)
        print(self.dataframe.isnull().sum().sort_values(ascending=False))
        print('#' * 20, 'Descriptive Statistics', '#' * 20)
        print(self.dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

    # categorical, numerical and cardinal variables
    def grab_col_names(self, cat_th=10, car_th=20):
        # cat_cols, cat_but_car
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in self.dataframe.columns if self.dataframe[col].nunique() < cat_th and
                       self.dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in self.dataframe.columns if self.dataframe[col].nunique() > car_th and
                       self.dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {self.dataframe.shape[0]}")
        print(f"Variables: {self.dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        
        return cat_cols, num_cols, cat_but_car

    # summary of categorical variables
    def cat_summary(self, cat_cols, plot=False):
        print("Categorical Columns Summary:")
        summary_list = []
        for col in cat_cols:
            counts = self.dataframe[col].value_counts()
            percentages = counts / len(self.dataframe) * 100
            df = pd.DataFrame({
                'Variable': col,
                'Category': counts.index,
                'Count': counts.values,
                'Percentage': percentages.values
            })
            summary_list.append(df)
        summary_df = pd.concat(summary_list, ignore_index=True)
        print(summary_df)

        if plot:
            n = len(cat_cols)
            fig_height = max(3 * n, 6)
            fig_width = 8
            font_size = min(max(10, 18 - n), 16)

            plt.rcParams.update({'font.size': font_size})

            fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height))
            if n == 1:
                axes = [axes]
            for i, col in enumerate(cat_cols):
                sns.countplot(data=self.dataframe, x=col, ax=axes[i], order=self.dataframe[col].value_counts().index)
                axes[i].set_xlabel(col.capitalize())
                axes[i].set_ylabel('Count')
                axes[i].set_title(f"{col.capitalize()} Bar Plot")
            plt.tight_layout()
            plt.show()
            plt.rcParams.update({'font.size': 12})  # Varsayılan font boyutuna geri dön

    # summary of numerical variables  
    def num_summary(self, num_cols, plot=False):
        print("Numerical Columns Summary:")
        print(self.dataframe[num_cols].describe().T)

        if plot:
            n = len(num_cols)
            # Dinamik olarak figür boyutu ve yazı boyutunu ayarla
            fig_height = max(4 * n, 6)
            fig_width = 12
            font_size = min(max(10, 18 - n), 16)  # n arttıkça küçült

            plt.rcParams.update({'font.size': font_size})

            fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height))
            if n == 1:
                axes = [axes]
            for i, col in enumerate(num_cols):
                sns.boxplot(data=self.dataframe, y=col, ax=axes[i][0])
                axes[i][0].set_xlabel(col.capitalize())
                axes[i][0].set_ylabel('Value')
                axes[i][0].set_title(f"{col.capitalize()} Boxplot")
                sns.histplot(data=self.dataframe, x=col, ax=axes[i][1])
                axes[i][1].set_xlabel(col.capitalize())
                axes[i][1].set_ylabel('Frequency')
                axes[i][1].set_title(f"{col.capitalize()} Histogram")
            plt.tight_layout()
            plt.show()
            plt.rcParams.update({'font.size': 12})  # Varsayılan font boyutuna geri dön

    # Analysis of target variable with categorical variables
    def target_summary_with_cat(self, target, cat_cols, cat_th=10, plot=False):
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in self.dataframe.columns if self.dataframe[col].nunique() < cat_th and
                       self.dataframe[col].dtypes != "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col != target]  # Hedef değişkeni hariç tut
        summary_list = []
        for col in cat_cols:
            # Her kategori için hedef değişkenin ortalaması ve sayısı
            stats = self.dataframe.groupby(col)[target].agg(['mean', 'count']).reset_index()
            stats['Variable'] = col
            stats = stats.rename(columns={'mean': f'{target}_mean', 'count': 'Count', col: 'Category'})
            summary_list.append(stats)
        summary_df = pd.concat(summary_list, ignore_index=True)
        # Kolon sırasını düzenle
        summary_df = summary_df[['Variable', 'Category', f'{target}_mean', 'Count']]
        print(summary_df)

        # Her kategorik değişken için barplot
        if plot:
            n = len(cat_cols)
            fig_height = max(3 * n, 6)
            fig_width = 8
            font_size = min(max(10, 18 - n), 16)
            plt.rcParams.update({'font.size': font_size})

            fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height))
            if n == 1:
                axes = [axes]
            for i, col in enumerate(cat_cols):
                sns.barplot(x=col, y=target, data=self.dataframe, ax=axes[i], ci=False)
                axes[i].set_xlabel(col.capitalize())
                axes[i].set_ylabel(f"{target.capitalize()} Mean")
                axes[i].set_title(f"{col.capitalize()} vs {target.capitalize()} Mean")
            plt.tight_layout()
            plt.show()
            plt.rcParams.update({'font.size': 12})  # Varsayılan font boyutuna geri dön

    # Analysis of target variable with numerical variables
    def target_summary_with_num(self, target, num_cols, plot=False):
        # Her değişken için mean, median, std, var hesapla
        summary_list = []
        for col in num_cols:
            stats = self.dataframe.groupby(target)[col].agg(['mean', 'median', 'std', 'var', 'min', 'max', 'count', 'skew', 'sem']).reset_index()
            stats['Variable'] = col
            summary_list.append(stats)
        summary_df = pd.concat(summary_list, ignore_index=True)
        # Kolon sırasını düzenle
        cols = [target, 'Variable', 'mean', 'median', 'std', 'var', 'min', 'max', 'count', 'skew', 'sem']
        summary_df = summary_df[cols]
        print(summary_df)

        # Tüm boxplot'ları tek figürde çiz
        if plot:
            n = len(num_cols)
            fig_height = max(4 * n, 6)
            fig_width = 8
            font_size = min(max(10, 18 - n), 16)
            plt.rcParams.update({'font.size': font_size})

            fig, axes = plt.subplots(n, 1, figsize=(fig_width, fig_height))
            if n == 1:
                axes = [axes]
            for i, col in enumerate(num_cols):
                sns.boxplot(x=self.dataframe[target], y=self.dataframe[col], ax=axes[i])
                axes[i].set_xlabel(target.capitalize())
                axes[i].set_ylabel(col.capitalize())
                axes[i].set_title(f"{col.capitalize()} by {target.capitalize()}")
            plt.tight_layout()
            plt.show()
            plt.rcParams.update({'font.size': 12})  # Varsayılan font boyutuna geri dön

    # high_correlated_cols' function to delete the variables with high correlation
    def high_correlated_cols(self, num_cols, corr_threshold=0.90, plot=False):
        corr = self.dataframe[num_cols].corr()
        cor_matrix = corr.abs()
        upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
        drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_threshold)]

        if plot:
            plt.figure(figsize=(max(10, len(num_cols)), max(8, len(num_cols))))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, cmap='RdBu', annot=True, linewidths=0.5, linecolor='w', mask=mask)
            plt.title(f'Correlation Matrix (threshold={corr_threshold})')
            plt.show()

        print(f"Highly correlated columns (>{corr_threshold}): {drop_list}")
        return drop_list, corr

    # DATA PREPROCESSING

    # finding outlier thresholds
    def outlier_thresholds(self, col_name, lower_quantile=0.05, upper_quantile=0.95):
        lower_quartile = self.dataframe[col_name].quantile(lower_quantile)
        upper_quartile = self.dataframe[col_name].quantile(upper_quantile)
        iqr = upper_quartile - lower_quartile
        upper_limit = round(upper_quartile + 1.5 * iqr, 2)
        lower_limit = round(lower_quartile - 1.5 * iqr, 2)
        return lower_limit, upper_limit

    # replace outliers with thresholds
    def replace_with_thresholds(self, variable, lower_quantile=0.05, upper_quantile=0.95):
        lower_quartile = self.dataframe[variable].quantile(lower_quantile)
        upper_quartile = self.dataframe[variable].quantile(upper_quantile)
        iqr = upper_quartile - lower_quartile
        upper_limit = round(upper_quartile + 1.5 * iqr, 2)
        lower_limit = round(lower_quartile - 1.5 * iqr, 2)
        self.dataframe.loc[(self.dataframe[variable] < lower_limit), variable] = lower_limit
        self.dataframe.loc[(self.dataframe[variable] > upper_limit), variable] = upper_limit

    # is there any outliers?
    def has_outliers(self, col_name, lower_quantile=0.05, upper_quantile=0.95):
        lower_quartile = self.dataframe[col_name].quantile(lower_quantile)
        upper_quartile = self.dataframe[col_name].quantile(upper_quantile)
        iqr = upper_quartile - lower_quartile
        upper_limit = round(upper_quartile + 1.5 * iqr, 2)
        lower_limit = round(lower_quartile - 1.5 * iqr, 2)
        return self.dataframe[
            ((self.dataframe[col_name] < lower_limit) | (self.dataframe[col_name] > upper_limit))].any(axis=None)

    # printing outliers
    def print_outliers(self, column_name, lower_quantile=0.05, upper_quantile=0.95, show_index=False):
        lower_quartile = self.dataframe[column_name].quantile(lower_quantile)
        upper_quartile = self.dataframe[column_name].quantile(upper_quantile)
        iqr = upper_quartile - lower_quartile
        upper_limit = round(upper_quartile + 1.5 * iqr, 2)
        lower_limit = round(lower_quartile - 1.5 * iqr, 2)
        if self.dataframe[
            ((self.dataframe[column_name] < lower_limit) | (self.dataframe[column_name] > upper_limit))].shape[0] > 10:
            print(self.dataframe[((self.dataframe[column_name] < lower_limit) | (
                        self.dataframe[column_name] > upper_limit))].head())
        else:
            print(self.dataframe[
                      ((self.dataframe[column_name] < lower_limit) | (self.dataframe[column_name] > upper_limit))])

        if show_index:
            outlier_index = self.dataframe[
                ((self.dataframe[column_name] < lower_limit) | (self.dataframe[column_name] > upper_limit))].index
            return outlier_index

    # removing outliers
    def remove_outliers(self, column_name, lower_quantile=0.05, upper_quantile=0.95):
        lower_quartile = self.dataframe[column_name].quantile(lower_quantile)
        upper_quartile = self.dataframe[column_name].quantile(upper_quantile)
        iqr = upper_quartile - lower_quartile
        upper_limit = round(upper_quartile + 1.5 * iqr, 2)
        lower_limit = round(lower_quartile - 1.5 * iqr, 2)
        df_without_outliers = self.dataframe[
            ~((self.dataframe[column_name] < lower_limit) | (self.dataframe[column_name] > upper_limit))]
        return df_without_outliers

    # missing values
    def missing_values_table(self, return_cols=False):
        missing_cols = self.dataframe.columns[self.dataframe.isnull().any()]
        missing_count = self.dataframe[missing_cols].isnull().sum().sort_values(ascending=False)
        missing_ratio = (missing_count / self.dataframe.shape[0]) * 100
        missing_data = pd.concat([missing_count, missing_ratio], axis=1, keys=['Missing Count', 'Missing Ratio (%)'])
        print(missing_data, end="\n")
        if return_cols:
            return missing_cols

    # missing values vs target
    def missing_vs_target(self, target, missing_cols):
        temp_df = self.dataframe.copy()
        for col in missing_cols:
            temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
        na_flags = temp_df.columns[temp_df.columns.str.contains("_NA_FLAG")]
        for col in na_flags:
            print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                                "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

    # label encoding
    def label_encoder(self, binary_col):
        labelencoder = LabelEncoder()
        self.dataframe[binary_col] = labelencoder.fit_transform(self.dataframe[binary_col])
        return self.dataframe

    # one-hot encoding
    def one_hot_encoder(self, categorical_cols, drop_first=True):
        self.dataframe = pd.get_dummies(self.dataframe, columns=categorical_cols, drop_first=drop_first)
        return self.dataframe

    # rare analyzer
    def rare_analyzer(self, target, categorical_columns, threshold=0.05):
        for col in categorical_columns:
            counts = self.dataframe[col].value_counts(normalize=True)
            rare_labels = counts[counts < threshold].index
            print(f'{col} : {len(rare_labels)}')
            print(pd.DataFrame({'COUNT': self.dataframe[col].value_counts(),
                                'RATIO': self.dataframe[col].value_counts(normalize=True),
                                'TARGET_MEAN': self.dataframe.groupby(col)[target].mean()}), end='\n\n\n')

    # encoding rare colums
    def rare_encoder(self, rare_perc=0.05):
        temp_df = self.dataframe.copy()
        rare_columns = [col for col in temp_df.columns if
                        temp_df[col].dtypes == 'O' and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(
                            axis=None)]
        for col in rare_columns:
            counts = temp_df[col].value_counts() / len(temp_df)
            rare_labels = counts[counts < rare_perc].index
            temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])
        return temp_df

    def fuzzy_closest_matches(self, word, column_name, limit):
        # get the top 10 closest matches to given word
        # convert to lower case
        temp_df = self.dataframe.copy()
        temp_df[column_name] = temp_df[column_name].str.lower()
        # remove trailing white spaces
        temp_df[column_name] = temp_df[column_name].str.strip()

        closest_matches = fuzzywuzzy.process.extract(word, temp_df[column_name].unique(), limit=limit,
                                                     scorer=fuzzywuzzy.fuzz.token_sort_ratio)
        return closest_matches

    def replace_matches_in_column(self, column, string_to_match, limit=10, min_ratio=45):
        # get a list of unique strings
        temp_df = self.dataframe.copy()

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

    def neattext_nlp(self, column, puncts=True, stopwords=True, urls=True, emails=True,
                     numbers=False, multiple_whitespaces=False, currency_symbols=True,
                     special_char=True):
        temp_df = self.dataframe.copy()
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


