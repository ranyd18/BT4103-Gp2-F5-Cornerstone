print("download data from snowflake")
import snowflake.connector

#create connection
con=snowflake.connector.connect(
    account='f5networks.east-us-2.azure',
    warehouse = 'PRD_ENT_TABLEAU_WH',
    database='PRD_ENT_RAW',
    schema='OBIA',
    authenticator='oauth',
    token="https://login.microsoftonline.com/dd3dfd2f-6a3b-40d1-9be0-bf8327d81c50/oauth2/v2.0/token",
    )

cur = con.cursor()
# Execute a statement that will generate a result set.
sql = "select * from t"
cur.execute(sql)
# Fetch the result set from the cursor and deliver it as the Pandas DataFrame.
df = cur.fetch_pandas_all()


print("import clean_data")
from pipeline_clean_data import column_selection, text_processing

print("column selection")
column_selection(df)

print("text_processing")
text_processing(df)

print("done cleaning data")

print("import clustering")
from pipeline_clustering import clustering
cluster_results = clustering(df['X_PRODUCT'].tolist(),df['PROCESSED_PAR'].tolist())
print("done clustering")

df['cluster'] = cluster_results
df.to_csv('with_cluster.csv')
print("done saving results as csv")

