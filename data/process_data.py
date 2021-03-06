import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load the input dataset where the messages are
    as .csv file with available messages and the
    categories_filepath point to a .csv files
    of the categories"""
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the two files into one on id column
    return messages.merge(categories, on="id")


def clean_data(df):
    """Clean the dataframe based on apriori knwoledge of the
    behaviour of the input types"""
    # Split the categories ino saperate category columns
    categories = df.categories.str.split(";", expand=True)
    # Cleanup the names
    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert the category values to numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df = df.drop(labels=["categories"], axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove undefined (2) values
    df = df[df.related != 2]

    return df


def save_data(df, database_filename):
    """
    Save the pandas dataframe into an sqlite datafile
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("frame1", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
