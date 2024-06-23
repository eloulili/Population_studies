import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from scipy.stats import differential_entropy as entropy

list_csv_names = ["AGG_CSV\AGG_CSV\Time_points_dfs_01.csv",  "AGG_CSV\AGG_CSV\Time_points_dfs_04.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_05.csv",  "AGG_CSV\AGG_CSV\Time_points_dfs_07.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_08.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_09.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_10.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_11.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_12.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_13.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_14.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_15.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_16.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_18.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_19.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_20.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_22.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_23.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_24.csv",  "AGG_CSV\AGG_CSV\Time_points_dfs_26.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_27.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_28.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_29.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_30.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_31.csv",
                   "AGG_CSV\AGG_CSV\Time_points_dfs_32.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_33.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_34.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_35.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_36.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_37.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_38.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_39.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_40.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_41.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_42.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_43.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_44.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_45.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_46.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_47.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_48.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_49.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_50.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_51.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_52.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_53.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_54.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_55.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_56.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_57.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_58.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_59.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_60.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_61.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_62.csv",
                   "AGG_CSV\AGG_CSV\Time_points_dfs_63.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_64.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_65.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_66.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_67.csv", "AGG_CSV\AGG_CSV\Time_points_dfs_68.csv"]


# Load the data
df_first_time_stamp = pd.DataFrame({
    'C1_mean': [],
    'area': [],
    "max_F_C2": [],
	"min_F_C2": [],
	"std_F_C2": [],
    "mean_F_C2": [],
    "Entropy_F_C2": [],
    "max_F_C3": [],
    "min_F_C3": [],
    "std_F_C3": [],
    "mean_F_C3": [],
    "Entropy_F_C3": [],
    "Number_cells": [],
    "Time_interval": [],
    "Time_Stamp": [],
    'n_simulations': []
})
df_last_time_stamp = pd.DataFrame({
    'C1_mean': [],
    'area': [],
    "max_F_C2": [],
	"min_F_C2": [],
	"std_F_C2": [],
    "mean_F_C2": [],
    "Entropy_F_C2": [],
    "max_F_C3": [],
    "min_F_C3": [],
    "std_F_C3": [],
    "mean_F_C3": [],
    "Entropy_F_C3": [],
    "Number_cells": [],
    "Time_interval": [],
    "Time_Stamp": [],
    'n_simulations': []
})

df_population_evolution = pd.DataFrame({
    'n_cells': [],
    'Time': []})


df_entropy_area_distribution = pd.DataFrame({"Entropy": [],
    "n_cells": [],
    'n_simulations': []})


df_entropy_radius_distribution = pd.DataFrame({"Entropy": [],
    "n_cells": [],
    'n_simulations': []})

df_total_entropy = pd.DataFrame({"Entropy": [],
    "n_cells": [],})

last_timestamps = []
total_traits_per_population = [[] for i in range(180)]


duration_array = []
n_cells_array = []
j = 0
start = time.time()
for csv_file in list_csv_names:
    location = f"..\{csv_file}"
    data = pd.read_csv(location, sep=",")

    Time_Stamp = data['Time_Stamp']
    Time_Stamp = Time_Stamp.unique()
    first_timestamp = Time_Stamp[0]
    n_cells = [sum(data['Time_Stamp'] == i) for i in Time_Stamp]
    artifacts_indexes = []
    last_non_artifact_index = 0
    for index in range(len(n_cells) - 1):
        if n_cells[index] - n_cells[last_non_artifact_index] > 50  :
            artifacts_indexes.append(index)
        else:
            last_non_artifact_index = index*1
    n_cells = np.delete(n_cells, artifacts_indexes)
    Time_Stamp = np.delete(Time_Stamp, artifacts_indexes)
    last_index = 0
    for i in range(len(n_cells)):
        if n_cells[i] > 200:
            last_index = i
            break
    last_timestamp = Time_Stamp[last_index]
    filtered_data = data[data['Time_Stamp'] == first_timestamp]
    filtered_data['n_simulations'] = [j] * len(filtered_data)
    df_first_time_stamp = pd.concat([df_first_time_stamp, filtered_data], ignore_index=True, axis=0)

    last_timestamps.append(last_timestamp)
    filtered_data = data[data['Time_Stamp'] == last_timestamp]
    filtered_data['n_simulations'] = [j] * len(filtered_data)
    df_last_time_stamp = pd.concat([df_last_time_stamp, filtered_data], ignore_index=True, axis=0)

    entropy_area_values = []
    entropy_radius_values = []
    n_cells_values = []
    for index in range(last_index):
        if index not in artifacts_indexes and n_cells[index] > 4:
            entropy_area_values.append(entropy(data[data['Time_Stamp'] == Time_Stamp[index]]['area']))
            entropy_radius_values.append(entropy(np.sqrt(data[data['Time_Stamp'] == Time_Stamp[index]]['area']/np.pi)))
            n_cells_values.append(n_cells[index])
    sims = [j] * len(entropy_area_values)
    df_entropy_area_distribution = pd.concat([df_entropy_area_distribution, pd.DataFrame({"Entropy": entropy_area_values, "n_cells": n_cells_values, 'n_simulations': sims})], ignore_index=True, axis=0)
    studied_populations_number = [False] * 180
    traits_to_add = []
    for population in range(20,200):
        has_broke = False
        for index in range(last_index):
            if index not in artifacts_indexes:
                if not studied_populations_number[population-20] and n_cells[index] == population:
                    studied_populations_number[population-20] = True
                    traits_to_add.append(list(data[data['Time_Stamp'] == Time_Stamp[index]]['area']))
                    has_broke = True
                    break
        if not has_broke:
            traits_to_add.append(None)
              


    durations = np.zeros(len(Time_Stamp))
    initial_time = datetime.datetime.strptime(Time_Stamp[0], '%Y%m%d-%H%M%S')
    for i in range(1,len(Time_Stamp)):
        timestamp_datetime = datetime.datetime.strptime(Time_Stamp[i], '%Y%m%d-%H%M%S')
        durations[i] = (timestamp_datetime - initial_time).total_seconds()
    duration_array.append(durations)
    n_cells_array.append(n_cells)

    unusable = False
    for population in range(20,200):
        if not studied_populations_number[population-20]:
            if studied_populations_number[population - 21]:
                traits_to_add[population -20] = traits_to_add[population - 21]
                studied_populations_number[population-20 ] = True
            else:
                for index in range(population - 20, 181):
                    if index == 180:
                        unusable = True
                        break
                    if studied_populations_number[index]:
                        traits_to_add[population - 20]=traits_to_add[index]
                        break
                studied_populations_number[population-20] = True
        if unusable:
            break
        total_traits_per_population[population - 20].extend(traits_to_add[population - 20])

    j += 1
    print("File number: ", j)

end = time.time()


print("Time taken to load the data: ", end - start)

start = time.time()
all_times = np.concatenate(duration_array)
all_times = np.sort(all_times)
all_times = np.unique(all_times)

all_cells = np.zeros(len(all_times))
for i in range(len(duration_array)):
    all_cells += np.interp(all_times, duration_array[i], n_cells_array[i], left=1, right=1)
total_entrop = []
for i in range(180):
    total_entrop.append(entropy(total_traits_per_population[i]))

total_entropy = pd.DataFrame({"Entropy": total_entrop, "n_cells": [20 + i for i in range(180)] })
all_cells = all_cells / len(duration_array)
pd.DataFrame(df_entropy_area_distribution).to_csv("entropy_area_values.csv", index=False)
pd.DataFrame(df_entropy_radius_distribution).to_csv("entropy_radius_values.csv", index=False)
pd.DataFrame(total_entropy).to_csv("Total_entropy.csv", index=False)
#pd.DataFrame(all_cells).to_csv("..\AGG_CSV\AGG_CSV\All_cells.csv", index=False)
#pd.DataFrame(all_times).to_csv("..\AGG_CSV\AGG_CSV\All_times.csv", index=False)
#pd.DataFrame(df_first_time_stamp).to_csv("..\AGG_CSV\AGG_CSV\First_time_stamp.csv", index=False)
#pd.DataFrame(df_last_time_stamp).to_csv("..\AGG_CSV\AGG_CSV\Last_time_stamp.csv", index=False)






# Plot the data

plt.plot(all_times, all_cells)
plt.xlabel('Time')
plt.ylabel('Number of cells')

plt.show()
