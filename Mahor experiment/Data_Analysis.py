# Import necessary libraries for data manipulation, numerical operations, plotting, and statistical analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from scipy.stats import differential_entropy as entropy

# List of columns relevant for the analysis
r = ['area','C1_mean', "mean_F_C2", "Entropy_F_C2" ,"std_F_C2",  "mean_F_C3", "Entropy_F_C3", "std_F_C3"]

# Set a limit for the last timestamp
last_time = 30000

# Load data from CSV files containing timestamps and traits of cells
first_time_stamp = pd.read_csv("..\AGG_CSV\AGG_CSV\First_time_stamp.csv", sep=",")
last_time_stamp = pd.read_csv("..\AGG_CSV\AGG_CSV\Last_time_stamp.csv", sep=",")
all_cells = pd.read_csv("..\AGG_CSV\AGG_CSV\All_cells.csv", sep=",")
all_times = pd.read_csv("..\AGG_CSV\AGG_CSV\All_times.csv", sep=",")
entropy_area_distribution = pd.read_csv("..\AGG_CSV\AGG_CSV\entropy_area_values.csv", sep=",")
entropy_radius_distribution = pd.read_csv("..\AGG_CSV\AGG_CSV\entropy_radius_values.csv", sep=",")
total_entropy = pd.read_csv("..\AGG_CSV\AGG_CSV\Total_entropy.csv", sep=",")

# Number of first generations to be considered
n_1 = 16

# Extract time stamps from the dataset
Time_Stamp = last_time_stamp['Time_Stamp']

# Determine the earliest time stamp and set it as initial time
initial_time = datetime.datetime.strptime(np.min(first_time_stamp['Time_Stamp']), '%Y%m%d-%H%M%S')


n_bins = 35
# Initialize list to store durations from initial time to each timestamp
duration = []
current_sim = None
for i in range(len(Time_Stamp)):
    if last_time_stamp['n_simulations'][i] == current_sim:
        pass
    current_sim = last_time_stamp['n_simulations'][i]
    timestamp_datetime = datetime.datetime.strptime(Time_Stamp[i], '%Y%m%d-%H%M%S')
    duration.append((timestamp_datetime - initial_time).total_seconds())

# Convert list of durations to a numpy array for further analysis
duration = np.array(duration)
print(f"min durations: {min(duration)}")
print(f"max durations: {max(duration)}")
print(f"mean durations: {np.mean(duration)}")
print(f"std durations: {np.std(duration)}")

# Initialize lists for storing cell counts and simulation results
all_times = np.array(all_times)
all_cells = np.array(all_cells)
n_initial_cells = []
end_n_cells = []
usable_all_times = []
usable_all_cells = []
prediction = []

# Filter and store data within specified time range and compute predictions
for i in range(len(all_times)):
    if 150 < all_times[i][0] < last_time:
        usable_all_times.append(all_times[i][0])
        usable_all_cells.append(all_cells[i][0])
        prediction.append(all_cells[0][0]* 2**((all_times[i][0]/3600 * 0.5)))

# Print the number of time steps processed
print(f"n_timesteps: {len(usable_all_times)}\n")

# Plotting actual and predicted cell counts
plt.plot(usable_all_times, usable_all_cells)
plt.plot(usable_all_times, prediction, label='prediction')  
plt.legend()
plt.title("Population ecolution")

for n_sim in range(1, 69):
    entropy_area_values = entropy_area_distribution[entropy_area_distribution['n_simulations'] == n_sim]['Entropy']
    entropy_radius_values = entropy_radius_distribution[entropy_radius_distribution['n_simulations'] == n_sim]['Entropy']
    n_cells_values = entropy_area_distribution[entropy_area_distribution['n_simulations'] == n_sim]['n_cells']
    plt.plot(n_cells_values, entropy_area_values, label=f"n_sim: {n_sim}")

plt.title("Evolution of entropy of area with population size in every chamber")

plt.figure()
Total_entropy = total_entropy['Entropy']
population = [20 + i for i in range(180)]
plt.plot(population, Total_entropy)
plt.title("Evolution of total entropy with population size")
    

# Define a function to calculate n1 based on given formula
def get_n1(n2, cv, ratio):
    return ((n2 -1)*(cv**2 + 1))/(ratio - 1)

def get_n1_entropy(n2, var, ratio):
    return np.exp(ratio * np.log(2*np.pi*n2*var)) / (2*np.pi*var) - n2

# Define a function to calculate expected entropy ratio
def get_expected_entopy_ratio(n_1, n_2, mean_var):
    return np.log(2*np.pi*(n_2+n_1)*mean_var)/ np.log(2*np.pi*n_2*mean_var)

# Process different traits and calculate various statistical measures
for trait_name in ['area','C1_mean', "mean_F_C2", "mean_F_C3" ]:

    if trait_name == 'area':
        # In order to compute the radius which is not in the original dataset
        Mean_initial_trait= [0] * 69

        i = 0
        n_cells = 0
        Sum_trait = 0 
        current_sim = first_time_stamp['n_simulations'][0]
        n_initial_cells = []

        # Compute the mean of the trait for the first time stamp
        for index, cell in first_time_stamp.iterrows(): 
            if cell['n_simulations'] == current_sim:
                n_cells += 1
                Sum_trait += np.sqrt(cell[trait_name]/np.pi)
            else:
                Mean_initial_trait[int(current_sim)] = (Sum_trait / n_cells)
                current_sim = cell['n_simulations']
                Sum_trait = np.sqrt(cell[trait_name]/np.pi)
                n_initial_cells.append(n_cells)
                n_cells = 1


        filtered_list = last_time_stamp[last_time_stamp['n_simulations'] != 68] # Remove the last time stamp which is problematic
        
        End_trait = list(np.sqrt(filtered_list[trait_name]/np.pi))
        Total_variance = np.var(End_trait)
        Total_entropy = entropy(End_trait)
        max_trait = np.max(End_trait)
        min_trait = np.min(End_trait)


        Variance_per_simulation = []
        Entropy_per_simulation = []
        Current_simulation_trait = []
        Mean_final_trait = []
        CV = [] # Coefficient of variation
        current_sim = first_time_stamp['n_simulations'][0]
        end_n_cells = []
        Ratios = []
        Ratios_entropy = []

        for index, cell in last_time_stamp.iterrows():
            if cell['n_simulations'] == current_sim:
                Current_simulation_trait.append(np.sqrt(cell[trait_name]/np.pi))
            else:
                Variance_per_simulation.append(np.var(Current_simulation_trait))
                Entropy_per_simulation.append(entropy(Current_simulation_trait))
                end_n_cells.append(len(Current_simulation_trait))
                Mean_final_trait.append(np.mean(Current_simulation_trait))
                Ratios.append(Total_variance/Variance_per_simulation[-1] )
                Ratios_entropy.append(Total_entropy/Entropy_per_simulation[-1] )
                CV.append(np.std(Current_simulation_trait) / np.mean(Current_simulation_trait))
                if False and current_sim%1 == 0:
                    plt.figure()
                    plt.hist(Current_simulation_trait, bins = 20, color='r')
                    plt.title(f"Histogram of {trait_name} : lineage {current_sim}")
                Current_simulation_trait = [np.sqrt(cell[trait_name]/np.pi)]
                current_sim = cell['n_simulations']
        mean_cv = np.mean(CV)
        Mean_variance = np.mean(Variance_per_simulation)
        
        mean_ratio = np.mean(Ratios)
        median_ratio = np.median(Ratios)
        mean_ratio_v2 = Total_variance/np.mean(Variance_per_simulation)

        # Compute the number of generations in the first step based on the formula
        # Using the ratio of the total variance over the mean variance
        n_1_from_ratio_on_mean_variance = get_n1(np.log2(np.mean(end_n_cells)), mean_cv,  Total_variance/np.mean(Variance_per_simulation) )
        # Using the mean of the ratios
        n_1_from_mean_of_ratios = get_n1(np.log2(np.mean(end_n_cells)), mean_cv,  np.mean(Ratios) )
        
        expected_ratios = []
        for i in range(len(end_n_cells)):
            # The expected ratio is calculated based on the formula for each chamber
            expected_ratios.append(1 + ((np.log2(end_n_cells[i]) -1) / n_1 )* (1 + CV[i]**2))   
        mean_expected_ratio = np.mean(expected_ratios)

        # The expected ratio is calculated based on the formula for the whole population
        global_expected_ratio = 1 + ((np.log2(np.mean(end_n_cells)) -1) / n_1) * (1 + mean_cv**2)
        std_ratios= np.std(Ratios)
        
        
        
        n_bins = np.histogram_bin_edges(End_trait, bins='sturges')
        print(f"n_bins: {len(n_bins)}")
        Hist = np.zeros(len(n_bins))
        for i in range(len(End_trait)):
            Hist[np.digitize(End_trait[i], n_bins)-1] += 1
        Hist = Hist / len(End_trait)
        shanon_entropy = -np.sum(Hist * np.log(Hist))

        plt.figure()
        plt.hist(End_trait, bins = n_bins, density=False )
        plt.axvline(x=np.mean(Mean_initial_trait), color='b', linestyle='dashed', linewidth=2)
        plt.title(f"Histogram of radius : all poopulation")

        bins = np.histogram_bin_edges(Ratios, bins='sturges')
        plt.figure()
        plt.hist(Ratios, bins = bins, density=False, color='g' )
        plt.axvline(x=mean_ratio, color='r', linestyle='dashed', linewidth=2, label='mean_ratio')
        plt.axvline(x=mean_expected_ratio, color='b', linestyle='dashed', linewidth=2, label='mean_expected_ratio')
        plt.title(f"Histogram of Ratios : {trait_name}")
        plt.legend()

        print(f"Max_trait: {max_trait}")
        print(f"Min_trait: {min_trait}")
        print(f'Mean_variance : {Mean_variance}')
        print(f'Total_variance : {Total_variance}')
        print(f"n_1_from_mean_of_ratios : {n_1_from_mean_of_ratios}")
        print(f"n_1_from_ratio_on_mean_variance : {n_1_from_ratio_on_mean_variance}")
        print(f"CV: {mean_cv}")
        print(f"mean_ratio: {mean_ratio}")
        print(f"Median ratio: {median_ratio}")
        print(f"std of ratio: {std_ratios}")
        print(f"mean_expected_ratio: {mean_expected_ratio}")
        print(f"global_expected_ratio: {global_expected_ratio}  \n")

        print(f"Sannon entropy: {shanon_entropy} \n")
  
        print(f"Mean initial radius :"  + f" {np.mean(Mean_initial_trait)}\n \n")
        print("############################ \n \n \n")
        



############################################################################################################
    """
    Same as before but for the other traits, without manipulating the data to compute the radius 
    """

    Mean_initial_trait= [0] * 69

    i = 0
    n_cells = 0
    Sum_trait = 0 
    current_sim = first_time_stamp['n_simulations'][0]
    n_initial_cells = []
    for index, cell in first_time_stamp.iterrows(): 
        if cell['n_simulations'] == current_sim:
            n_cells += 1
            Sum_trait += cell[trait_name]
        else:
            Mean_initial_trait[int(current_sim)]=(Sum_trait / n_cells)
            current_sim = cell['n_simulations']
            Sum_trait = cell[trait_name]
            n_initial_cells.append(n_cells)
            n_cells = 1
            if current_sim == 68:
                break


    filtered_list = last_time_stamp[last_time_stamp['n_simulations'] != 68]
    End_trait = list(filtered_list[trait_name])
    Total_variance = np.var(End_trait)
    Total_entropy = entropy(End_trait)
    max_trait = np.max(End_trait)
    min_trait = np.min(End_trait)

    Variance_per_simulation = []
    Entropy_per_simulation = []
    Current_simulation_trait = []
    Mean_final_trait = []
    CV = []
    current_sim = first_time_stamp['n_simulations'][0]
    end_n_cells = []
    Ratios = []
    Ratios_entropy = []
    Computed_n1 = []

    for index, cell in last_time_stamp.iterrows():
            if cell['n_simulations'] == current_sim:
                Current_simulation_trait.append(cell[trait_name])
            else:
                Variance_per_simulation.append(np.var(Current_simulation_trait))
                Entropy_per_simulation.append(entropy(Current_simulation_trait))
                end_n_cells.append(len(Current_simulation_trait))
                Mean_final_trait.append(np.mean(Current_simulation_trait))
                Ratios.append(Total_variance/Variance_per_simulation[-1] )
                Ratios_entropy.append(Total_entropy/Entropy_per_simulation[-1] )
                CV.append(np.std(Current_simulation_trait) / np.mean(Current_simulation_trait))
                Computed_n1.append(get_n1(np.log2(end_n_cells[-1]), CV[-1], Ratios[-1]))
                if current_sim%30 == 0 and False:
                    # To plot the distribution in one chamber
                    plt.figure()
                    plt.hist(Current_simulation_trait, bins = 20, color='r')
                    plt.title(f"Histogram of {trait_name} : lineage {current_sim}")
                Current_simulation_trait = [cell[trait_name]]
                current_sim = cell['n_simulations']

    Mean_variance = np.mean(Variance_per_simulation)               
    mean_cv = np.mean(CV)
    mean_ratio = np.mean(Ratios)
    median_ratio = np.median(Ratios)
    mean_ratio_v2 = Total_variance/np.mean(Variance_per_simulation)
    n_1_from_ratio_on_mean_variance = get_n1(np.log2(np.mean(end_n_cells)), mean_cv,  Total_variance/np.mean(Variance_per_simulation) )
    n_1_from_mean_of_ratios = get_n1(np.log2(np.mean(end_n_cells)), mean_cv,  np.mean(Ratios) )
    n_1_from_mean_of_ratios_entropy = get_n1(np.log2(np.mean(end_n_cells)), mean_cv,  np.mean(Ratios_entropy) )
      
    expected_ratios = []
    expected_ratios_entropy = []

    for i in range(len(end_n_cells)):
        expected_ratios.append(1 + ((np.log2(end_n_cells[i]) -1) / n_1 )* (1 + CV[i]**2))
        expected_ratios_entropy.append(get_expected_entopy_ratio(n_1, np.log2(end_n_cells[i]), Mean_variance))
    mean_expected_ratio = np.mean(expected_ratios)
    global_expected_ratio = 1 + ((np.log2(np.mean(end_n_cells)) -1) / n_1) * (1 + mean_cv**2)
    std_ratios= np.std(Ratios)
    
    n_bins = np.histogram_bin_edges(End_trait, bins='sturges')
    print(f"n_bins: {len(n_bins)}")

    Hist = np.zeros(len(n_bins))
    for i in range(len(End_trait)):
            Hist[np.digitize(End_trait[i], n_bins)-1] += 1
    Hist = Hist / len(End_trait)
    shanon_entropy = -np.sum(Hist * np.log(Hist))

    plt.figure()
    plt.hist(End_trait, bins = n_bins, density=False )
    plt.axvline(x=np.mean(Mean_initial_trait), color='b', linestyle='dashed', linewidth=2)
    plt.title(f"Histogram of {trait_name}: all poopulation")

    bins = np.histogram_bin_edges(Ratios, bins='sturges')
    plt.figure()
    plt.hist(Ratios, bins = bins, density=False, color='g' )
    plt.axvline(x=mean_ratio, color='r', linestyle='dashed', linewidth=2, label='mean_ratio')
    plt.axvline(x=mean_expected_ratio, color='b', linestyle='dashed', linewidth=2, label='mean_expected_ratio')
    plt.title(f"Histogram of Ratios : {trait_name}")
    plt.legend()

    print(f"Max_trait: {max_trait}")
    print(f"Min_trait: {min_trait}")
    print(f'Mean_variance : {Mean_variance}')
    print(f'Total_variance : {Total_variance}')
    print(f"n_1_from_mean_of_ratios : {n_1_from_mean_of_ratios}")
    print(f"n_1_from_ratio_on_mean_variance : {n_1_from_ratio_on_mean_variance}")
    print(f"CV: {mean_cv}")
    print(f"mean_ratio: {mean_ratio}")
    print(f"Median ratio: {median_ratio}")
    print(f"std of ratio: {std_ratios}")
    print(f"mean_expected_ratio: {mean_expected_ratio}")
    print(f"global_expected_ratio: {global_expected_ratio}  \n")

    print(f"Sannon entropy: {shanon_entropy} \n")
  
    print(f"Mean initial radius :"  + f" {np.mean(Mean_initial_trait)}\n \n")
    print("############################ \n \n \n")
    


print("n_initial_cells: ", n_initial_cells)
print("mean initial cells: ", np.mean(n_initial_cells))
print("end_n_cells: ", end_n_cells)
print("mean end cells: ", np.mean(end_n_cells))
    

n_generations = []
for i in range(len(n_initial_cells)):
        n_generations.append(np.log2(end_n_cells[i] / n_initial_cells[i]))

print("n_generations: ", n_generations)
plt.show()
