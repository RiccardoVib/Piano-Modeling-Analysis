import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def make_plots(data_dir_s, name, proj, c, type):
    if type == 3:
        with plt.style.context('seaborn-whitegrid'):
            fig = plt.figure(figsize=(6, 6))
            ax = plt.axes(projection="3d")
            ax.scatter3D(proj[:182, 0],
                 proj[:182, 1],
                 proj[:182, 2],
                 label='Acoustic',
                 c=c[0])
            ax.scatter3D(proj[182:182+174, 0],
                         proj[182:182+174, 1],
                         proj[182:182+174, 2],
                         label='Sample-based',
                         c=c[1])
            ax.scatter3D(proj[182+174:, 0],
                         proj[182+174:, 1],
                         proj[182+174:, 2],
                         label='Physic-based',
                         c=c[2])
            ax.set_xlabel('1st PCA Component')
            ax.set_ylabel('2nd PCA Component')
            ax.set_zlabel('3nd PCA Component')
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(data_dir_s + name + '_3D.png')
            plt.close('all')
    else:
        with plt.style.context('seaborn-whitegrid'):
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(proj[:182, 0],
                         proj[:182, 1],
                         label='Acoustic',
                         c=c[0])
            plt.scatter(proj[182:182+174, 0],
                        proj[182:182+174, 1],
                        label='Sample-based',
                        c=c[1])
            plt.scatter(proj[182+174:, 0],
                        proj[182+174:, 1],
                        label='Physic-based',
                        c=c[2])
            plt.xlabel('1st PCA Component')
            plt.ylabel('2nd PCA Component')
            plt.legend(loc='upper right')
            plt.tight_layout()

        fig.savefig(data_dir_s + name + '_2D.png')
        plt.close('all')
        return

def LDA(f, labels, name, c):
    lda = LinearDiscriminantAnalysis()
    lda.fit(f, labels)
    # this is only for scatter plot purposes
    projected_features = lda.transform(f)
    data_dir_s = '../../../Analysis/Figs/'

    plt.rcParams['font.size'] = 16

    # Plot the distribution of the data according to the first two principle components
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(6, 6))
        plt.title('Silhouette coefficient: 0.75')
        plt.scatter(projected_features[:182, 0],
                        projected_features[:182, 1],
                        label='Acoustic',
                        c=c[0])
        plt.scatter(projected_features[182:182 + 174, 0],
                        projected_features[182:182 + 174, 1],
                        label='PCM-based',
                        c=c[1])
        plt.scatter(projected_features[182 + 174:, 0],
                        projected_features[182 + 174:, 1],
                        label='Physics-based',
                        c=c[2])
        plt.xlabel('1st LDA Component')
        plt.ylabel('2nd LDA Component')
        plt.legend(loc='lower right', fontsize="13")
        plt.tight_layout()
        #plt.show()
        fig.savefig(data_dir_s + name + '_LDA.pdf', format='pdf')
        plt.close('all')
    return projected_features