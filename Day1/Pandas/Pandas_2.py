import pandas as pd

def main():
  try:
      df = pd.read_csv('C:\\Users\\HP\\AI with Python\\Day1\\data.csv')  #Replace with your file path


      df = df.drop(['LastUpdated'], axis=1)


      df.rename(columns={'Stock': 'StockLeft',}, inplace=True)


      df.columns = df.columns.str.strip()


      print(df.head(10))

  except FileNotFoundError:
        print(f"Error: The file 'C:\\Users\\HP\\AI with Python\\Day1\\data.csv' was not found.")


if __name__ == "__main__":
    main()