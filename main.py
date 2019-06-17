import warnings

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

n_clusters = 5

blob, blob_y = make_blobs(n_samples=30, random_state=0)
blob2, blob2_y = make_blobs(n_samples=300, random_state=0)
blob3, blob3_y = make_blobs(n_samples=900, random_state=361)
circle, circle_y = make_circles(n_samples=30, noise=0.17, random_state=0)
circle2, circle2_y = make_circles(n_samples=300, noise=0.17, random_state=0)
circle3, circle3_y = make_circles(n_samples=900, noise=0.17, random_state=361)
moon, moon_y = make_moons(n_samples=30, noise=0.09, random_state=0)
moon2, moon2_y = make_moons(n_samples=300, noise=0.09, random_state=0)
moon3, moon3_y = make_moons(n_samples=900, noise=0.09, random_state=361)

data = [(blob, blob_y), (blob2, blob2_y), (blob3, blob3_y), (circle, circle_y), (circle2, circle2_y),
        (circle3, circle3_y), (moon, moon_y), (moon2, moon2_y), (moon3, moon3_y)]

wyniki = open('wyniki.txt', 'w')

for count, (dat, dat_y) in enumerate(data):
    wyniki.write("Dane nr " + str(count) + ":\n")
    X, y = dat, dat_y

    X = StandardScaler().fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], s=5)
    plt.savefig("wyniki\dane" + str(count) + "\dane" + str(count) + ".png")
    plt.clf()

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_

    wyniki.write(
        "Wskaźnik Silhouette'a algorytmu K-Means dla danych nr " + str(
            count) + ": " + "%.4f\n" % silhouette_score(
            X, labels).round(4))
    wyniki.write(
        "Skorygowany indeks Randa algorytmu K-Means danych nr " + str(count) + ": " + "%.4f\n\n" % adjusted_rand_score(y,
                                                                                                                 labels))

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=7, c='black')
    plt.savefig("wyniki\dane" + str(count) + "\KMeans" + str(count) + ".png")
    plt.clf()

    # Agglomerative Clustering

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglomerative.fit_predict(X)

    labels = agglomerative.labels_

    wyniki.write("Wskaźnik Silhouette'a algorytmu Agglomerative Clustering danych nr " + str(
        count) + ": " + "%.4f\n" % silhouette_score(X, labels).round(
        4))
    wyniki.write("Skorygowany indeks Randa algorytmu Agglomerative Clustering danych nr " + str(
        count) + ": " + "%.4f\n\n" % adjusted_rand_score(y, labels))

    plt.scatter(X[:, 0], X[:, 1], c=agglomerative.labels_, s=5)
    plt.savefig("wyniki\dane" + str(count) + "\Agglomerative" + str(count) + ".png")
    plt.clf()

    # Spectral

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

    labels = spectral.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=5)
    plt.savefig("wyniki\dane" + str(count) + "\Spectral" + str(count) + ".png")
    plt.clf()
    wyniki.write("Wskaźnik Silhouette'a algorytmu Spectral Clustering danych nr " + str(
        count) + ": " + "%.4f\n" % silhouette_score(X, labels).round(4))
    wyniki.write("Skorygowany indeks Randa algorytmu Spectral Clustering danych nr " + str(
        count) + ": " + "%.4f\n\n" % adjusted_rand_score(y, labels))

    # Gaussian
    gaussian = GaussianMixture(n_components=n_clusters, covariance_type='full')

    labels = gaussian.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=5)
    plt.savefig("wyniki\dane" + str(count) + "\Gauusian" + str(count) + ".png")
    plt.clf()

    wyniki.write("Wskaźnik Silhouette'a algorytmu Gaussian Mixture danych nr " + str(
        count) + ": " + "%.4f\n" % silhouette_score(X, labels).round(4))
    wyniki.write("Skorygowany indeks Randa algorytmu Gaussian Mixture danych nr " + str(
        count) + ": " + "%.4f\n\n" % adjusted_rand_score(y, labels))

wyniki.close()
