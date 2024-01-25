import pandas as pd

# Assuming your DataFrame is named 'df' and has a column 'earnings'
# Create a sample DataFrame for illustration
data = {'earnings': [5000, 3000, 7000, 2000]}
df = pd.DataFrame(data)

# Add a new column 'earnings_rank' based on the sorting order of 'earnings'
df['earnings_rank'] = df['earnings'].rank().astype(int)

# If you want integer values instead of floating-point, you can convert them
# df['earnings_rank'] = df['earnings_rank'].astype(int)

# Display the DataFrame
print(df)
