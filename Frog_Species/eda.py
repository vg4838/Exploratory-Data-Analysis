import pandas as pd
from pylab import *
import numpy as np


def scatter_frog_ss(frog_ss):
    # Plotting raw features for frog
    groups = frog_ss.groupby("Species")
    for name, group in groups:
        plt.plot(group["MFCCs_10"], group["MFCCs_17"], marker="o", linestyle="", label=name)
    plt.legend()
    plt.title('Frog Subsample')
    plt.xlabel("MFCCs_10")
    plt.ylabel("MFCCs_17")
    plt.show()
    # plt.savefig("FrogSS_scatter.png")


def scatter_frog(frog):
    # Plotting raw features for frog
    groups = frog.groupby("Species")
    for name, group in groups:
        plt.plot(group["MFCCs_10"], group["MFCCs_17"], marker="o", linestyle="", label=name, markersize=1.5)
    plt.legend()
    plt.title('Frog')
    plt.xlabel("MFCCs_10")
    plt.ylabel("MFCCs_17")
    plt.show()
    # plt.savefig("Frog_scatter.png")


def hist_frog_ss(x1_fss, y1_fss, x2_fss, y2_fss):
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.6, hspace=1)
    fig.suptitle('Frog Subsample')
    axs[0, 0].hist(x1_fss)
    axs[0, 0].set_title('')
    axs[0, 1].hist(y1_fss)
    axs[0, 1].set_title('')
    axs[1, 0].hist(x2_fss)
    axs[1, 0].set_title('')
    axs[1, 1].hist(y2_fss)
    axs[1, 1].set_title('')
    xaxis = ['HylaMinuta', 'HylaMinuta', 'HypsiboasCinerascens', 'HypsiboasCinerascens']
    yaxis = ['MFCCs_10', 'MFCCs_17', 'MFCCs_10', 'MFCCs_17']
    for i, ax in enumerate(axs.flat):
        ax.set(xlabel=xaxis[i], ylabel=yaxis[i])
    plt.show()
    # plt.savefig('FrogSS_hist.png')


def hist_frog(x1_f, y1_f, x2_f, y2_f):
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.6, hspace=1)
    fig.suptitle('Frog')
    axs[0, 0].hist(x1_f)
    axs[0, 1].hist(y1_f)
    axs[1, 0].hist(x2_f)
    axs[1, 1].hist(y2_f)
    xaxis = ['HylaMinuta', 'HylaMinuta', 'HypsiboasCinerascens', 'HypsiboasCinerascens']
    yaxis = ['MFCCs_10', 'MFCCs_17', 'MFCCs_10', 'MFCCs_17']
    for i, ax in enumerate(axs.flat):
        ax.set(xlabel=xaxis[i], ylabel=yaxis[i])
    plt.show()
    # plt.savefig('Frog_hist.png')


def line_frog_ss(x1_fss, y1_fss, x2_fss, y2_fss):
    x1_fss = x1_fss.sort_values().reset_index(drop=True)
    y1_fss = y1_fss.sort_values().reset_index(drop=True)
    x2_fss = x2_fss.sort_values().reset_index(drop=True)
    y2_fss = y2_fss.sort_values().reset_index(drop=True)
    df = pd.DataFrame({'HM_MFCCs_10': x1_fss, 'HM_MFCCs_17': y1_fss, 'HC_MFCCs_10': x2_fss, 'HC_MFCCs_17': y2_fss})
    plt.title('Frog SubSample')
    plt.plot(df)
    plt.legend(('HM_MFCCs_10', 'HM_MFCCs_17', 'HC_MFCCs_10', 'HC_MFCCs_17'))
    plt.show()
    # plt.savefig('FrogSS_Line.png')


def line_frog(x1_f, y1_f, x2_f, y2_f):
    x1_f = x1_f.sort_values().reset_index(drop=True)
    y1_f = y1_f.sort_values().reset_index(drop=True)
    x2_f = x2_f.sort_values().reset_index(drop=True)
    y2_f = y2_f.sort_values().reset_index(drop=True)
    df1 = pd.DataFrame({'HM_MFCCs_10': x1_f, 'HM_MFCCs_17': y1_f, 'HC_MFCCs_10': x2_f, 'HC_MFCCs_17': y2_f})
    plt.plot(df1)
    plt.title('Frog')
    plt.legend(('HM_MFCCs_10', 'HM_MFCCs_17', 'HC_MFCCs_10', 'HC_MFCCs_17'))
    plt.show()
    # plt.savefig('Frog_Line.png')


def box_frog_ss(x1_fss, y1_fss, x2_fss, y2_fss):
    plt.boxplot([x1_fss, y1_fss, x2_fss, y2_fss])
    plt.xticks([1, 2, 3, 4], ['HM_MFCCs_10', 'HM_MFCCs_17', 'HC_MFCCs_10', 'HC_MFCCs_17'])
    plt.title('Frog Subsample')
    plt.show()
    # plt.savefig('FrogSS_box.png')


def box_frog(x1_f, y1_f, x2_f, y2_f):
    plt.boxplot([x1_f, y1_f, x2_f, y2_f])
    plt.xticks([1, 2, 3, 4], ['HM_MFCCs_10', 'HM_MFCCs_17', 'HC_MFCCs_10', 'HC_MFCCs_17'])
    plt.title('Frog')
    plt.show()
    # plt.savefig('Frog_box.png')


def bar_error_ss(x1_fss, y1_fss, x2_fss, y2_fss):
    x1_fss_mean = np.mean(x1_fss)
    y1_fss_mean = np.mean(y1_fss)
    x2_fss_mean = np.mean(x2_fss)
    y2_fss_mean = np.mean(y2_fss)
    x1_fss_std = np.std(x1_fss)
    y1_fss_std = np.std(y1_fss)
    x2_fss_std = np.std(x2_fss)
    y2_fss_std = np.std(y2_fss)
    xaxis = ['HM_MFCCs_10', 'HM_MFCCs_17', 'HC_MFCCs_10', 'HC_MFCCs_17']
    x_pos = np.arange(len(xaxis))
    CTEs = [x1_fss_mean, y1_fss_mean, x2_fss_mean, y2_fss_mean]
    error = [x1_fss_std, y1_fss_std, x2_fss_std, y2_fss_std]
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='red', capsize=5)
    ax.set_ylabel('Frequency band intensities')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xaxis)
    ax.set_title('Bar plot with errors for Frog subsample')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.show()
    # plt.savefig('FrogSS_bar_plot_with_error_bars.png')


def bar_error(x1_f, y1_f, x2_f, y2_f):
    x1_f_mean = np.mean(x1_f)
    y1_f_mean = np.mean(y1_f)
    x2_f_mean = np.mean(x2_f)
    y2_f_mean = np.mean(y2_f)
    x1_f_std = np.std(x1_f)
    y1_f_std = np.std(y1_f)
    x2_f_std = np.std(x2_f)
    y2_f_std = np.std(y2_f)
    xaxis = ['HM_MFCCs_10', 'HM_MFCCs_17', 'HC_MFCCs_10', 'class2_attr2']
    x_pos = np.arange(len(xaxis))
    CTEs = [x1_f_mean, y1_f_mean, x2_f_mean, y2_f_mean]
    error = [x1_f_std, y1_f_std, x2_f_std, y2_f_std]
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='red', capsize=5)
    ax.set_ylabel('Frequency band intensities')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xaxis)
    ax.set_title('Bar plot with errors for Frog')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('Frog_bar_plot_with_error_bars.png')
    plt.show()


def desc_frog_ss(frog_ss):
    print("\nFrog subsample statistics:\n")
    print("------------------------------------")
    a1_mean = np.mean(frog_ss['MFCCs_10'])
    print(f"MFCCs_10 mean {a1_mean}")
    a2_mean = np.mean(frog_ss['MFCCs_17'])
    print(f"MFCCs_17 mean {a2_mean}")
    a1_sd = np.std(frog_ss['MFCCs_10'])
    print(f"MFCCs_10 std {a1_sd}")
    a2_sd = np.std(frog_ss['MFCCs_17'])
    print(f"MFCCs_17 std {a2_sd}")
    data = np.array([frog_ss['MFCCs_10'], frog_ss['MFCCs_17']])
    print("\nCovariance matrix for MFCCs_10 and MFCCs_17\n")
    covMatrix = np.cov(data, bias=True)
    print(covMatrix)


def desc_frog(frog):
    print("\nFrog dataset statistics:\n")
    print("------------------------------------")
    a1_mean = np.mean(frog['MFCCs_10'])
    print(f"MFCCs_10 mean {a1_mean}")
    a2_mean = np.mean(frog['MFCCs_17'])
    print(f"MFCCs_17 mean {a2_mean}")
    a1_sd = np.std(frog['MFCCs_10'])
    print(f"MFCCs_10 std {a1_sd}")
    a2_sd = np.std(frog['MFCCs_17'])
    print(f"MFCCs_17 std {a2_sd}")
    data = np.array([frog['MFCCs_10'], frog['MFCCs_17']])
    print("\nCovariance matrix for MFCCs_10 and MFCCs_17\n")
    covMatrix = np.cov(data, bias=True)
    print(covMatrix)


def main():
    path = input('Enter file path until the folder:\n')
    frog_ss = pd.read_csv(path + 'Frogs-subsample.csv')
    frog = pd.read_csv(path + 'Frogs.csv')
    scatter_frog_ss(frog_ss)
    scatter_frog(frog)
    frog_c1 = frog_ss.iloc[np.where(frog_ss['Species'] == 'HylaMinuta')]
    frog_c2 = frog_ss.iloc[np.where(frog_ss['Species'] == 'HypsiboasCinerascens')]
    x1_fss = frog_c1['MFCCs_10']
    y1_fss = frog_c1['MFCCs_17']
    x2_fss = frog_c2['MFCCs_10']
    y2_fss = frog_c2['MFCCs_17']
    hist_frog_ss(x1_fss, y1_fss, x2_fss, y2_fss)
    frog_c1 = frog.iloc[np.where(frog['Species'] == 'HylaMinuta')]
    frog_c2 = frog.iloc[np.where(frog['Species'] == 'HypsiboasCinerascens')]
    x1_f = frog_c1['MFCCs_10']
    y1_f = frog_c1['MFCCs_17']
    x2_f = frog_c2['MFCCs_10']
    y2_f = frog_c2['MFCCs_17']
    hist_frog(x1_f, y1_f, x2_f, y2_f)
    line_frog_ss(x1_fss, y1_fss, x2_fss, y2_fss)
    line_frog(x1_f, y1_f, x2_f, y2_f)
    box_frog_ss(x1_fss, y1_fss, x2_fss, y2_fss)
    box_frog(x1_f, y1_f, x2_f, y2_f)
    bar_error_ss(x1_fss, y1_fss, x2_fss, y2_fss)
    bar_error(x1_f, y1_f, x2_f, y2_f)
    desc_frog_ss(frog_ss)
    desc_frog(frog)


if __name__ == "__main__":
    main()
