from chembl_webresource_client.new_client import new_client
import pandas as pd


activity = new_client.activity
activity_types = ['Ki', 'IC50', 'EC50']
max_records = 1000
fields = ['activity_id', 'assay_chembl_id', 'molecule_chembl_id', 'activity_type', 'value', 'units']


activities_data = []


for a in activity.filter(activity_type__in=activity_types).only(*fields):
    activities_data.append(a)
    if len(activities_data) >= max_records:
        break


activities_df = pd.DataFrame.from_records(activities_data)

print(activities_df.head())
activities_df.to_csv("chembl_activities_limited.csv", index=False)
print("Data exported to chembl_activities_limited.csv")
