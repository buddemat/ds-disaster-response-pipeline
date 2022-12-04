"""
.. module:: process_data
   :platform: Unix, Windows
   :synopsis: A module to load data, clean it and store it in sqlite.

.. moduleauthor:: Matthias Budde


"""
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str):
    """Loads message and category data from CSVs and merges it.

    :param str messages_filepath: The path of the messages CSV
    :param str categories_filepath: The path of the categories CSV
    :return: data merged on ID column
    :rtype: pd.DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df_merged_data = messages.merge(categories, on='id')
    return df_merged_data


def clean_data(df_original):
    """Cleans data (eliminate duplicates, re-encode features...).

    :param pd.Dataframe df_original: Input dataframe to be cleaned
    :return: Cleaned dataframe
    :rtype: pd.DataFrame
    """
    df_clean = df_original.copy()

    categories_split = df_clean['categories'].str.split(';', expand=True)
    categories_split['id'] = df_clean['id']

    first_row = categories_split.iloc[0, :-1]
    category_colnames = [c[0:-2] for c in first_row]
    category_colnames.append('id')

    categories_split.columns = category_colnames
    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].apply(
            lambda x: x[-1:] if column != 'id' else x)
        categories_split[column] = pd.to_numeric(categories_split[column])

    df_clean = df_clean.drop('categories', axis=1)
    df_clean = df_clean.merge(categories_split, on='id')
    # re-encode column 'related' to binary
    df_clean['related'] = df_clean['related'].apply(lambda x: 1 if x == 2 else x)
    df_clean = df_clean.drop_duplicates()

    return df_clean


def save_data(df_clean, database_filename):
    """Saves cleaned data into sqlite database.

    :param pd.Dataframe df_clean: The dataframe that should be saved into sqlite
    :param str database_filename: The name (path) of the sqlite db file
    :return:
    :rtype:
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df_clean.to_sql('CategorizedMessages', engine, index=False, if_exists='replace')


def main():
    """Main function.
    """
    if len(sys.argv) == 4:

        messages_filepath = sys.argv[1]
        categories_filepath = sys.argv[2]
        database_filepath = sys.argv[3]

        print('Loading data...\n'
              f'    MESSAGES: {messages_filepath}\n'
              f'    CATEGORIES: {categories_filepath}')
        df_data = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df_data = clean_data(df_data)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df_data, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
