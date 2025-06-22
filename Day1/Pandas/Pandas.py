import pandas as pd
def main():

    try:
        df = pd.read_csv('C:\\Users\\HP\\AI with Python\\Day1\\data.csv')

        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        print("\nLast 3 rows of the dataset:")
        print(df.tail(3))

    except FileNotFoundError:
        print(f"Error: The file 'C:\\Users\\HP\\AI with Python\\Day1\\data.csv' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file 'C:\\Users\\HP\\AI with Python\\Day1\\data.csv' is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file 'C:\\Users\\HP\\AI with Python\\Day1\\data.csv' is corrupted or has invalid content.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()