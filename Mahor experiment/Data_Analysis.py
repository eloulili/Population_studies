import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

r = ['area', 'C1_mean', 'mean_F_C2', 'mean_F_C3',]

first_time_stamp = pd.read_csv("AGG_CSV\AGG_CSV\First_time_stamp.csv", sep=",")
last_time_stamp = pd.read_csv("AGG_CSV\AGG_CSV\Last_time_stamp.csv", sep=",")
n_initial_cells = []
end_n_cells = []


def get_n1(n2, cv, ratio):
     return ((n2 -1)*(cv**2 + 1))/(ratio - 1)
     

for trait_name in ['area', 'C1_mean', 'mean_F_C2', 'mean_F_C3',]:
    Mean_initial_trait= []

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
            Mean_initial_trait.append(Sum_trait / n_cells)
            current_sim = cell['n_simulations']
            Sum_trait = cell[trait_name]
            n_initial_cells.append(n_cells)
            n_cells = 1


    End_trait = list(last_time_stamp[trait_name])
    Total_variance = np.var(End_trait)


    Variance_per_simulation = []
    Current_simulation_trait = []
    Mean_final_trait = []
    CV = []
    current_sim = None
    end_n_cells = []
    Ratios = []
    Computed_n1 = []
    for index, cell in last_time_stamp.iterrows():
        if cell['n_simulations'] == current_sim:
            Current_simulation_trait.append(cell[trait_name])
        else:
            Variance_per_simulation.append(np.var(Current_simulation_trait))
            end_n_cells.append(len(Current_simulation_trait))
            Mean_final_trait.append(np.mean(Current_simulation_trait))
            Ratios.append(Total_variance/Variance_per_simulation[-1] )
            CV.append(np.std(Current_simulation_trait) / np.mean(Current_simulation_trait))
            Computed_n1.append(get_n1(np.log2(end_n_cells[-1]), CV[-1], Ratios[-1]))
            Current_simulation_trait = []
            current_sim = cell['n_simulations']
    Variance_per_simulation.pop(0)
    end_n_cells.pop(0)
    Mean_final_trait.pop(0)
    CV.pop(0)
    Ratios.pop(0)
    Computed_n1.pop(0)
    mean_cv = np.mean(CV)
    mean_ratio = np.mean(Ratios)
    n_1p = get_n1(np.log2(np.mean(end_n_cells)), mean_cv,  Total_variance/np.mean(Variance_per_simulation) )

    
    
    Mean_variance = np.mean(Variance_per_simulation)

    print(f'Mean_variance : {Mean_variance}')
    print(f'Total_variance : {Total_variance}')
    #print(f"Ratio: {Ratios}")
    print(f"Computed_n1: {np.mean(Computed_n1)}, std: {np.std(Computed_n1)}")
    print(f"n_1p: {n_1p}")
    #print(f"n_1:{Computed_n1}")


    print(f"Mean initial :" + trait_name + f" {np.mean(Mean_initial_trait)}\n")


print("n_initial_cells: ", n_initial_cells)
print("end_n_cells: ", end_n_cells)
    

n_generations = []
for i in range(len(n_initial_cells)):
        n_generations.append(np.log2(end_n_cells[i] / n_initial_cells[i]))

print("n_generations: ", n_generations)
