from chembl_webresource_client.new_client import new_client
import pandas as pd
from time import time

columns = ['activity_id', 
           'assay_chembl_id', 
           'assay_description',
           'assay_type',
           'canonical_smiles',
           'document_chembl_id',
           'document_journal',
           'document_year',
           'molecule_chembl_id',
           'parent_molecule_chembl_id',
           'standard_type',
           'standard_units',
           'standard_value',
          ]

def format_seconds_to_hh_mm_ss(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

active_data = new_client.activity
active_data = active_data.filter(standard_type__in=["IC50", "EC50", "Ki"])
active_data = active_data.filter(standard_relation__exact=["="])


t0 = time()
t1 = time()
total = len(active_data)

with open("output.csv", "w") as f:
    line = "index"
    for column in columns:
        line += "\t"
        line += column
    line += "\n"
    f.write(line)
    for i, data in enumerate(active_data.only(columns)):
        line = str(i)
        for column in columns:
            line += "\t"
            line += str(data[column])
        line += "\n"
        f.write(line)
        if ((i+1)%1000) == 0:
            current = time()
            timepoint = format_seconds_to_hh_mm_ss(current - t0)
            remaining = format_seconds_to_hh_mm_ss((((current-t1)*((total-i)/1000))))
            print(f"{timepoint}: {(i/total)*100:.2f}% completed -- remaining: {remaining}")
            t1 = current
