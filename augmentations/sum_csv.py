import pandas as pd

path = "/media/disk2/proj_embedding_aug/extracted_mag40x_patch256_fp/process_list_autogen.csv"

df2=pd.read_csv(path)
total_patches = df2['bag_size'].sum()
total_aug = 10
total_features = 1024

print(len(df2["slide_id"]))
print("total patches:", total_patches)
print("total embeddings:", total_patches * total_aug * total_features)