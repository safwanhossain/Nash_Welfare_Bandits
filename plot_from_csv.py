#!/usr/bin/python3
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

def read_from_csv(filename):
    """ Indicies must be 4 elements """
    t_vals, eps_t, explore_ratio, mean_regrets, cum_regrets, std_regrets= [], [], [], [], [], []
    csv_file = open(filename, mode='r')
    csv_reader = csv.reader(csv_file, delimiter=",")
        
    for row in csv_reader:
        scale = 1.5
        t_vals.append(float(row[0]))
        eps_t.append(float(row[1]))
        explore_ratio.append(float(row[2]))
        mean_regrets.append(float(row[3]))
        cum_regrets.append(float(row[4]))
        std_regrets.append(float(row[5]))
    
    return np.array(t_vals), np.array(eps_t), np.array(explore_ratio), \
            np.array(mean_regrets), np.array(cum_regrets), np.array(std_regrets)

def plot(t_vals, mean_regrets, cum_regrets, std_regrets, plt_name, plot_var=False):
    fig, ax1 = plt.subplots()
    labels = ["Cumulative Regret", "Mean Regret"]
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel(labels[0], color=colors[0])
    ax1.plot(t_vals, cum_regrets, color=colors[0], label=labels[0], alpha=0.75)
    if plot_var:
        ax1.fill_between(t_vals, cum_regrets-std_regrets, cum_regrets+std_regrets,\
                color=colors[0], alpha=0.35)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(labels[1], color=colors[1])
    
    ax2.plot(t_vals, mean_regrets, color=colors[1], label=labels[1], alpha=0.75)
    if plot_var:
        ax1.fill_between(t_vals, mean_regrets-std_regrets, mean_regrets+std_regrets,\
                color=colors[1], alpha=0.35)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(plt_name)

if __name__ == "__main__":
    """ Run this file as "./plot_from_csv <csv_file_name>
    """
    inp_file = sys.argv[1]
    plt_name = inp_file[:-3] + "png"
    print("Plotting saved to", plt_name)
    t_vals, eps_t, explore_ratio, mean_regrets, cum_regrets, std_regrets = \
            read_from_csv(inp_file)
    plot(t_vals, mean_regrets, cum_regrets, std_regrets, plt_name)
    
    fig, ax = plt.subplots()
    ax.plot(t_vals, explore_ratio, label="Exploration")
    ax.set_ylabel("Explore Ratio")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    explore_name = plt_name[:-4]+"_explore.png"
    print("Plotting saved to", explore_name)
    plt.savefig(explore_name)

