# Compares the column headers of two data frames
def compare_column_headers(df_1, df_2):
    headers_1 = list(df_1.columns.values)
    headers_2 = list(df_2.columns.values)
    try:
        for col in range(len(headers_1)):
            assert headers_1[col] == headers_2[col]
    except AssertionError:
        print("AssertionError: column names do not match up")

# Compares the row headers of two data frames
def compare_row_headers(df_1, df_2):
    headers_1 = list(df_1.index.values)
    headers_2 = list(df_2.index.values)
    try:
        for col in range(len(headers_1)):
            assert headers_1[col] == headers_2[col]
    except AssertionError:
        print("AssertionError: row names do not match up")