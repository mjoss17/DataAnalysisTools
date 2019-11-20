"""
EDA.py
Author: Matt Joss

This file generates univariate, bivariate and multivariate plots for
properly formatted multi class data. 

To generate the plots, use the run function and see the examples
for the avila, glass, and iris datasets. 
"""
import random
random.seed(123)
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Single_Models as a
import Utils as u
import PreProcess_Avila as av
import PreProcess_Glass as gl
import PreProcess_Iris as ir
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# MULTIVARIATE PLOTS
def multivariate(output, df_in, params_in, labels_in, name_of_dataset):
    with PdfPages(output) as pdf_pages:

        #Creating PCA Model 'pca_df' with params_pca
        scaler = StandardScaler() 
        st_data = scaler.fit_transform(params_in)
        pca_model = PCA(.90)
        pca_model.fit(st_data)
        pca_data = pca_model.transform(st_data)
        ex_variance = np.var(pca_data, axis=0)
        ex_variance_ratio = ex_variance / np.sum(ex_variance)

        #Creating LDA Model 'lda_df' with params_lda
        lda = LinearDiscriminantAnalysis(n_components=6)
        lda_data = lda.fit(params_in, labels_in).transform(params_in)

        # PCA Plot
        Xax = pca_data[:, 0]
        Yax = pca_data[:, 1]
        fig, ax = plt.subplots(figsize=(15, 12))
        fig.patch.set_facecolor('white')
        for l in np.unique(labels_in):
            ix = np.where(labels_in == l)
            ax.scatter(Xax[ix[0]], Yax[ix[0]], s=40, label=l)
        plt.xlabel("First Principal Component (" + str(round(100*ex_variance_ratio[0], 1)) + "%)")
        plt.ylabel("Second Principal Component (" + str(round(100*ex_variance_ratio[1], 1)) + "%)")
        plt.title("PCA of " + name_of_dataset + " Dataset")
        plt.legend() 
        pdf_pages.savefig(fig)

        # LDA Plot
        Xax = lda_data[:, 0]
        Yax = lda_data[:, 1]
        fig, ax = plt.subplots(figsize=(15, 12))
        for i in np.unique(labels_in):
            ix = np.where(labels_in == i)
            ax.scatter(Xax[ix[0]], Yax[ix[0]], alpha=.8, label=i)
        plt.xlabel('LD0')
        plt.ylabel('LD1')
        plt.title('LDA of ' + name_of_dataset + ' Dataset')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        pdf_pages.savefig(fig)



# BIVARIATE PLOTS
def bivariate(output, df_in, params_in, labels_in, name_of_dataset):
    with PdfPages(output) as pdf_pages:

    #heatmap
        fig = plt.figure(figsize=(12,10)) 
        ax = sns.heatmap(params_in.corr(), annot=True, cmap= 'cubehelix_r', xticklabels=params_in.columns,
            yticklabels=params_in.columns)
        plt.yticks(rotation=0)
        plt.title("Correlation Plot for " + name_of_dataset + " Data")
        pdf_pages.savefig(fig)

    # Bivariate combined histograms 
        for feat in list(params_in.columns):
            sns.set_style("white")
            data = []
            for l in np.unique(labels_in):
                data.append(df_in.loc[df_in['target'] == l][feat])
            num_bins = 50
            np.seterr(divide='ignore', invalid='ignore')
            width = (max(df_in[feat]) - min(df_in[feat])) / num_bins
            bins = []
            for i in range(0, num_bins):
                bins.append(min(df_in[feat]) + i*width)
            kwargs = dict(alpha=0.5, bins=bins, stacked=True)
            fig = plt.figure(figsize=(10,7))
            for i in range(len(np.unique(labels_in))):
                r = lambda: random.randint(0,255)
                plt.hist(data[i], **kwargs, color=('#%02X%02X%02X' % (r(),r(),r())))  
            plt.xlim(min(df_in[feat]),max(df_in[feat]))
            plt.gca().set(title=('Frequency of ' + feat + ' for All Classes'), ylabel='Frequency', xlabel='Value')
            plt.legend(title='Classes', loc='upper right')
            pdf_pages.savefig(fig)

        #bivariate boxplots
        for feat in list(params_in.columns):
            fig = plt.figure()
            ax = sns.boxplot(x="target", y=feat, data=df_in)
            plt.xlabel = np.unique(labels_in)
            ax.set_title("Boxplots for " + feat)
            pdf_pages.savefig(fig)

        # Bivariate Scatterplots for Data
        done = []
        for x in list(params_in.columns):
            done.append(x)
            for y in list(params_in.columns):
                if (x != y and y not in done):
                    print(x, " ", y)
                    sns.set(rc={"lines.linewidth": 0.4})
                    plot = sns.pairplot(x_vars=[str(x)], y_vars=[str(y)], data = df_in, hue="target", size=6)
                    plot.fig.suptitle("Scatterplot of " + x + " and " + y)
                    plot._legend.set_bbox_to_anchor((1.0, 0.22))
                    pdf_pages.savefig(plot.fig)
                    # with regression
                    plot = sns.pairplot(x_vars=[str(x)], y_vars=[str(y)], data = df_in, hue="target", size=6, kind="reg")
                    plot.fig.suptitle("Scatterplot of " + x + " and " + y + " with All Regressions")
                    plot._legend.set_bbox_to_anchor((1.0, 0.22))
                    pdf_pages.savefig(plot.fig)
                    # with single regression 
                    plot = sns.pairplot(x_vars=[str(x)], y_vars=[str(y)], data = df_in, size=6, kind="reg")
                    plot.fig.suptitle("Scatterplot of " + x + " and " + y + " with Single Regression")
                    pdf_pages.savefig(plot.fig)

    
# # UNIVARIATE 
def univariate(output, df_in, params_in, labels_in, name_of_dataset):
    with PdfPages(output) as pdf_pages:

        fig = plt.figure(figsize =(8,5))
        ax = fig.add_subplot(111)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.set_facecolor('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        txt = ""
        for l in np.unique(labels_in):
            txt += "(" + str(l) +") \n"
            label = df_in.loc[df_in['target'] == l]
            txt += "Count: " + str(len(label)) + "\n\n"
        plt.text(.01,0.01,txt , size=6)
        pdf_pages.savefig(fig)


        for feat in list(params_in.columns):
            print("FEAT " +feat)
            num_bins = 40
            r = lambda: random.randint(0,255)
            color = ('#%02X%02X%02X' % (r(),r(),r()))
            x = df_in[feat]
            # print("X " + str(x))
            pavg = u.normalitytests(x)
            fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
            axs.hist(x, bins=num_bins, color=color)
            axs.set_xlabel("Value")
            axs.set_ylabel("Frequency")
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            plt.title("Histogram of " + str(feat) + " (Norm p-val= " + str(pavg) + ")")
            pdf_pages.savefig(fig)
            
            fig = plt.figure()
            ax = sns.boxplot(x=feat, data=df_in, color=color)
            ax.set_title("Boxplot of " + str(feat))
            pdf_pages.savefig(fig)


def run():
    # df, params, labels = av.get_all_data()
    # multivariate("TESTING_EDA_avila_M.pdf", df, params, labels, "Avila")
    # bivariate("TESTING_EDA_avila_B.pdf", df, params, labels, "Avila")
    # univariate("TESTING_EDA_avila_U.pdf", df, params, labels, "Avila")

    # df, params, labels = gl.get_all_data()
    # multivariate("TESTING_EDA_glass_M.pdf", df, params, labels, "Glass")
    # bivariate("TESTING_EDA_glass_B.pdf", df, params, labels, "Glass")
    # univariate("TESTING_EDA_glass_U.pdf", df, params, labels, "Glass")
    
    df, params, labels = ir.get_all_data()
    multivariate("TESTING_EDA_iris_M.pdf", df, params, labels, "Iris")
    bivariate("TESTING_EDA_iris_B.pdf", df, params, labels, "Iris")
    univariate("TESTING_EDA_iris_U.pdf", df, params, labels, "Iris")

run()