#open csv as dataframe
def get_data(filename="historical_data.csv"):
    historical_data = pd.read_csv(filename)
    historical_data.set_index("closetime", inplace=True)
    return historical_data