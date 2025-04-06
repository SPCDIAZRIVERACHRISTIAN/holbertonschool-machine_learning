
## Clustering

### Description
0. Initialize K-meansmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef initialize(X, k):that initializes cluster centroids for K-means:Xis anumpy.ndarrayof shape(n, d)containing the dataset that will be used for K-means clusteringnis the number of data pointsdis the number of dimensions for each data pointkis a positive integer containing the number of clustersThe cluster centroids should be initialized with amultivariate uniform distributionalong each dimension ind:The minimum values for the distribution should be the minimum values ofXalong each dimension indThe maximum values for the distribution should be the maximum values ofXalong each dimension indYou should usenumpy.random.uniformexactly onceYou are not allowed to use any loopsReturns: anumpy.ndarrayof shape(k, d)containing the initialized centroids for each cluster, orNoneon failurealexa@ubuntu-xenial:0x01-clustering$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
initialize = __import__('0-initialize').initialize

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
    print(initialize(X, 5))
alexa@ubuntu-xenial:0x01-clustering$ ./0-main.py[[14.54730144 13.46780434]
 [20.57098466 33.55245039]
 [ 9.55556506 51.51143281]
 [48.72458008 20.03154959]
 [25.43826106 60.35542243]]
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:0-initialize.pyHelp×Students who are done with "0. Initialize K-means"Review your work×Correction of "0. Initialize K-means"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:README.md file exists and is not emptyFile existsFirst line contains#!/usr/bin/env python3No loops allowedNot allowed to import anything exceptimport numpy as npCorrect output: NormalCorrect output: High dimensionalXCorrect output: invalidXCorrect output: invalidkpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×0. Initialize K-meansCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---6/6pts

1. K-meansmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef kmeans(X, k, iterations=1000):that performs K-means on a dataset:Xis anumpy.ndarrayof shape(n, d)containing the datasetnis the number of data pointsdis the number of dimensions for each data pointkis a positive integer containing the number of clustersiterationsis a positive integer containing the maximum number of iterations that should be performedIf no change in the cluster centroids occurs between iterations, your function should returnInitialize the cluster centroids using a multivariate uniform distribution (based on0-initialize.py)If a cluster contains no data points during the update step, reinitialize its centroidYou should usenumpy.random.uniformexactly twiceYou may use at most 2 loopsReturns:C, clss, orNone, Noneon failureCis anumpy.ndarrayof shape(k, d)containing the centroid means for each clusterclssis anumpy.ndarrayof shape(n,)containing the index of the cluster inCthat each data point belongs toalexa@ubuntu-xenial:0x01-clustering$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./0-main.py
[[ 9.92511389 25.73098987]
 [30.06722465 40.41123947]
 [39.62770705 19.89843487]
 [59.22766628 29.19796006]
 [20.0835633  69.81592298]]Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:1-kmeans.pyHelp×Students who are done with "1. K-means"Review your work×Correction of "1. K-means"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as npMaximum of two loops allowedCorrect output: NormalCorrect output: cluster needs to be reinitializedCorrect output: variable cluster sizeCorrect output: early stoppingCorrect output: high dimensionalXCorrect output: invalidXCorrect output: invalidkCorrect output: invaliditerationspycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×1. K-meansCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---10/10pts

2. VariancemandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef variance(X, C):that calculates the total intra-cluster variance for a data set:Xis anumpy.ndarrayof shape(n, d)containing the data setCis anumpy.ndarrayof shape(k, d)containing the centroid means for each clusterYou are not allowed to use any loopsReturns:var, orNoneon failurevaris the total variancealexa@ubuntu-xenial:0x01-clustering$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    for k in range(1, 11):
        C, _ = kmeans(X, k)
        print('Variance with {} clusters: {}'.format(k, variance(X, C).round(5)))
alexa@ubuntu-xenial:0x01-clustering$ ./2-main.py
Variance with 1 clusters: 157927.7052
Variance with 2 clusters: 82095.68297
Variance with 3 clusters: 34784.23723
Variance with 4 clusters: 23158.40095
Variance with 5 clusters: 7868.52123
Variance with 6 clusters: 7406.93077
Variance with 7 clusters: 6930.66361
Variance with 8 clusters: 6162.15884
Variance with 9 clusters: 5843.92455
Variance with 10 clusters: 5727.41124
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:2-variance.pyHelp×Students who are done with "2. Variance"Review your work×Correction of "2. Variance"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as npNo loops allowedCorrect output: NormalCorrect output: High dimensionalXCorrect output: invalidXCorrect output: invalidCpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×2. VarianceCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---6/6pts

3. Optimize kmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef optimum_k(X, kmin=1, kmax=None, iterations=1000):that tests for the optimum number of clusters by variance:Xis anumpy.ndarrayof shape(n, d)containing the data setkminis a positive integer containing the minimum number of clusters to check for (inclusive)kmaxis a positive integer containing the maximum number of clusters to check for (inclusive)iterationsis a positive integer containing the maximum number of iterations for K-meansThis function should analyzeat least2 different cluster sizesYou should use:kmeans = __import__('1-kmeans').kmeansvariance = __import__('2-variance').varianceYou may use at most 2 loopsReturns:results, d_vars, orNone, Noneon failureresultsis a list containing the outputs of K-means for each cluster sized_varsis a list containing the difference in variance from the smallest cluster size for each cluster sizealexa@ubuntu-xenial:0x01-clustering$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    print(results)
    print(np.round(d_vars, 5))
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./3-main.py
[(array([[31.78625503, 37.01090945]]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0])), (array([[34.76990289, 28.71421162],
       [20.14417812, 69.38429903]]), array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0])), (array([[49.55185774, 24.76080087],
       [20.0835633 , 69.81592298],
       [19.8719982 , 32.85851127]]), array([2, 2, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 2, 0, 0, 1, 2, 2,
       1, 0, 2, 2, 0, 0, 2, 0, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 0, 2, 2,
       0, 2, 2, 2, 1, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0,
       2, 0, 1, 0, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 0, 2, 0, 1, 2, 0, 2, 2,
       0, 1, 2, 2, 0, 2, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0,
       2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 2, 2,
       0, 1, 0, 2, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0,
       1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 2, 0, 0, 1,
       2, 0, 0, 2, 0, 0, 2, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0,
       2, 2, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
       1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0,
       2, 1, 0, 2, 2, 0, 2, 0])), (array([[39.57566544, 20.48452248],
       [20.0835633 , 69.81592298],
       [19.62313956, 33.02895961],
       [59.22766628, 29.19796006]]), array([2, 2, 1, 2, 3, 2, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 0, 3, 0, 1, 2, 2,
       1, 3, 2, 2, 0, 0, 2, 3, 3, 2, 1, 2, 0, 2, 3, 2, 3, 1, 1, 0, 2, 2,
       0, 2, 2, 2, 1, 3, 3, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3,
       2, 0, 1, 3, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 3, 2, 3, 1, 2, 0, 2, 2,
       3, 1, 2, 2, 3, 2, 2, 1, 2, 1, 0, 3, 2, 2, 3, 0, 0, 2, 2, 0, 3, 0,
       2, 2, 0, 1, 0, 2, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 1, 3, 1, 2, 2, 2,
       3, 1, 3, 2, 2, 0, 2, 3, 2, 3, 1, 3, 1, 1, 3, 2, 2, 1, 3, 2, 3, 0,
       1, 1, 1, 3, 3, 0, 2, 2, 2, 0, 0, 3, 0, 2, 1, 1, 1, 1, 2, 0, 3, 1,
       2, 3, 3, 2, 0, 0, 2, 1, 2, 1, 0, 0, 1, 1, 3, 0, 3, 2, 0, 0, 3, 3,
       2, 2, 3, 0, 0, 0, 3, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
       1, 0, 0, 2, 1, 0, 3, 0, 3, 1, 1, 0, 0, 3, 0, 3, 0, 2, 2, 3, 3, 3,
       2, 1, 0, 2, 2, 0, 2, 0])), (array([[30.06722465, 40.41123947],
       [59.22766628, 29.19796006],
       [ 9.92511389, 25.73098987],
       [20.0835633 , 69.81592298],
       [39.62770705, 19.89843487]]), array([0, 0, 3, 0, 1, 2, 0, 4, 3, 4, 4, 0, 3, 4, 3, 2, 4, 1, 4, 3, 2, 0,
       3, 1, 0, 2, 4, 4, 2, 1, 1, 0, 3, 2, 4, 0, 1, 0, 1, 3, 3, 4, 0, 2,
       4, 0, 0, 0, 3, 1, 1, 3, 0, 3, 0, 2, 0, 2, 0, 0, 2, 2, 1, 2, 1, 1,
       0, 4, 3, 1, 3, 4, 0, 0, 2, 0, 3, 2, 0, 3, 1, 0, 1, 3, 2, 4, 2, 0,
       1, 3, 0, 2, 1, 0, 0, 3, 2, 3, 4, 1, 0, 2, 1, 4, 4, 0, 2, 4, 1, 4,
       2, 2, 4, 3, 4, 2, 2, 3, 2, 2, 4, 4, 0, 3, 0, 3, 3, 1, 3, 2, 0, 2,
       1, 3, 1, 2, 0, 4, 0, 1, 0, 1, 3, 1, 3, 3, 1, 2, 2, 3, 1, 2, 1, 4,
       3, 3, 3, 1, 1, 4, 2, 2, 0, 4, 4, 1, 4, 2, 3, 3, 3, 3, 0, 4, 1, 3,
       2, 1, 1, 2, 4, 4, 2, 3, 2, 3, 4, 4, 3, 3, 1, 4, 1, 0, 4, 4, 1, 1,
       2, 0, 1, 4, 4, 0, 1, 3, 1, 2, 0, 1, 0, 2, 0, 2, 2, 0, 0, 2, 3, 0,
       3, 4, 4, 2, 3, 4, 1, 4, 1, 3, 3, 4, 4, 1, 4, 1, 4, 0, 2, 1, 1, 1,
       2, 3, 4, 0, 2, 4, 2, 4])), (array([[44.18492017, 16.98881789],
       [30.06722465, 40.41123947],
       [38.18858711, 20.81726128],
       [59.22766628, 29.19796006],
       [20.0835633 , 69.81592298],
       [ 9.92511389, 25.73098987]]), array([1, 1, 4, 1, 3, 5, 1, 2, 4, 0, 2, 1, 4, 2, 4, 5, 2, 3, 2, 4, 5, 1,
       4, 3, 1, 5, 2, 2, 5, 3, 3, 1, 4, 5, 2, 1, 3, 1, 3, 4, 4, 2, 1, 5,
       0, 1, 1, 1, 4, 3, 3, 4, 1, 4, 1, 5, 1, 5, 1, 1, 5, 5, 3, 5, 3, 3,
       1, 2, 4, 3, 4, 2, 1, 1, 5, 1, 4, 5, 1, 4, 3, 1, 3, 4, 5, 2, 5, 1,
       3, 4, 1, 5, 3, 1, 1, 4, 5, 4, 2, 3, 1, 5, 3, 2, 2, 1, 5, 2, 3, 2,
       5, 5, 2, 4, 2, 5, 5, 4, 5, 5, 0, 2, 1, 4, 1, 4, 4, 3, 4, 5, 1, 5,
       3, 4, 3, 5, 1, 0, 1, 3, 1, 3, 4, 3, 4, 4, 3, 5, 5, 4, 3, 5, 3, 0,
       4, 4, 4, 3, 3, 0, 5, 5, 1, 0, 2, 3, 2, 5, 4, 4, 4, 4, 1, 0, 3, 4,
       5, 3, 3, 5, 2, 2, 5, 4, 5, 4, 2, 2, 4, 4, 3, 0, 3, 1, 0, 2, 3, 3,
       5, 1, 3, 2, 2, 1, 3, 4, 3, 5, 1, 3, 1, 5, 1, 5, 5, 1, 1, 5, 4, 1,
       4, 2, 2, 5, 4, 2, 3, 2, 3, 4, 4, 0, 2, 3, 0, 3, 2, 1, 5, 3, 3, 3,
       5, 4, 2, 1, 5, 2, 5, 2])), (array([[26.23935735, 39.56231098],
       [33.86397001, 36.21416257],
       [32.10246392, 43.52452575],
       [39.78587939, 19.72783208],
       [59.22766628, 29.19796006],
       [20.0835633 , 69.81592298],
       [ 9.92511389, 25.73098987]]), array([0, 1, 5, 2, 4, 6, 1, 3, 5, 3, 3, 2, 5, 3, 5, 6, 3, 4, 3, 5, 6, 0,
       5, 4, 1, 6, 3, 3, 6, 4, 4, 0, 5, 6, 3, 2, 4, 1, 4, 5, 5, 3, 2, 6,
       3, 0, 0, 1, 5, 4, 4, 5, 2, 5, 0, 6, 2, 6, 2, 1, 6, 6, 4, 6, 4, 4,
       0, 3, 5, 4, 5, 3, 0, 2, 6, 1, 5, 6, 2, 5, 4, 2, 4, 5, 6, 3, 6, 0,
       4, 5, 0, 6, 4, 2, 0, 5, 6, 5, 3, 4, 2, 6, 4, 3, 3, 0, 6, 3, 4, 3,
       6, 6, 3, 5, 3, 6, 6, 5, 6, 6, 3, 3, 0, 5, 1, 5, 5, 4, 5, 6, 0, 6,
       4, 5, 4, 6, 0, 3, 0, 4, 2, 4, 5, 4, 5, 5, 4, 6, 6, 5, 4, 6, 4, 3,
       5, 5, 5, 4, 4, 3, 6, 6, 2, 3, 3, 4, 3, 6, 5, 5, 5, 5, 0, 3, 4, 5,
       6, 4, 4, 6, 3, 3, 6, 5, 6, 5, 1, 3, 5, 5, 4, 3, 4, 2, 3, 3, 4, 4,
       6, 2, 4, 3, 3, 1, 4, 5, 4, 6, 1, 4, 0, 6, 0, 6, 6, 2, 2, 6, 5, 0,
       5, 3, 3, 6, 5, 3, 4, 3, 4, 5, 5, 3, 3, 4, 3, 4, 3, 0, 6, 4, 4, 4,
       6, 5, 3, 1, 6, 3, 6, 3])), (array([[46.45917139, 23.16813158],
       [17.46350686, 68.10438494],
       [59.60504708, 29.19922401],
       [32.37545229, 35.93762723],
       [38.66481538, 19.34921236],
       [28.96634723, 42.29059653],
       [ 9.92511389, 25.73098987],
       [24.01364794, 72.38323004]]), array([5, 3, 1, 5, 0, 6, 3, 4, 7, 4, 4, 5, 1, 4, 1, 6, 4, 2, 4, 1, 6, 5,
       7, 2, 3, 6, 4, 4, 6, 2, 2, 3, 7, 6, 4, 3, 2, 3, 2, 1, 7, 4, 5, 6,
       0, 5, 5, 3, 7, 2, 2, 1, 5, 7, 5, 6, 5, 6, 5, 3, 6, 6, 2, 6, 2, 2,
       5, 4, 1, 2, 1, 4, 5, 5, 6, 3, 7, 6, 5, 1, 2, 5, 2, 7, 6, 0, 6, 5,
       2, 1, 3, 6, 2, 5, 5, 7, 6, 7, 4, 2, 5, 6, 2, 4, 0, 5, 6, 4, 2, 4,
       6, 6, 4, 1, 4, 6, 6, 7, 6, 6, 0, 4, 5, 7, 3, 7, 1, 2, 1, 6, 3, 6,
       2, 1, 2, 6, 5, 4, 5, 2, 5, 2, 7, 2, 1, 7, 2, 6, 6, 1, 2, 6, 2, 4,
       7, 1, 1, 2, 2, 0, 6, 6, 5, 0, 4, 2, 0, 6, 7, 1, 1, 7, 5, 4, 2, 7,
       6, 2, 0, 6, 4, 4, 6, 1, 6, 1, 3, 4, 7, 1, 2, 4, 2, 5, 4, 4, 2, 2,
       6, 5, 2, 4, 4, 3, 2, 1, 2, 6, 3, 2, 3, 6, 5, 6, 6, 5, 5, 6, 1, 5,
       1, 4, 4, 6, 1, 4, 2, 4, 2, 1, 1, 0, 4, 2, 4, 2, 4, 5, 6, 2, 2, 2,
       6, 1, 4, 3, 6, 4, 6, 4])), (array([[41.41465903, 21.62570842],
       [37.61417321, 17.19733029],
       [26.6124459 , 43.91604964],
       [16.74238916, 69.20086704],
       [ 9.92511389, 25.73098987],
       [24.00581119, 70.53794517],
       [31.42335817, 38.75662838],
       [55.10816891, 27.75253127],
       [61.16625328, 29.87816185]]), array([6, 6, 3, 6, 7, 4, 6, 1, 5, 0, 1, 2, 3, 1, 3, 4, 1, 7, 0, 3, 4, 2,
       3, 7, 6, 4, 1, 0, 4, 8, 8, 6, 5, 4, 1, 6, 8, 6, 7, 3, 5, 0, 2, 4,
       0, 2, 6, 6, 5, 8, 7, 3, 2, 5, 6, 4, 6, 4, 6, 6, 4, 4, 8, 4, 8, 8,
       2, 1, 3, 8, 3, 0, 6, 6, 4, 6, 5, 4, 6, 3, 7, 6, 8, 5, 4, 0, 4, 2,
       7, 3, 6, 4, 7, 6, 6, 5, 4, 5, 0, 8, 6, 4, 8, 0, 0, 6, 4, 0, 8, 1,
       4, 4, 0, 3, 0, 4, 4, 5, 4, 4, 0, 0, 2, 5, 6, 5, 3, 8, 3, 4, 6, 4,
       8, 3, 7, 4, 2, 0, 2, 7, 2, 8, 5, 8, 3, 5, 8, 4, 4, 3, 7, 4, 7, 0,
       5, 3, 5, 8, 7, 0, 4, 4, 6, 0, 1, 8, 0, 4, 5, 3, 3, 5, 6, 1, 8, 5,
       4, 8, 7, 4, 1, 0, 4, 5, 4, 3, 6, 0, 5, 3, 8, 1, 8, 6, 1, 1, 8, 8,
       4, 6, 8, 1, 0, 6, 8, 3, 8, 4, 6, 8, 6, 4, 6, 4, 4, 6, 2, 4, 5, 2,
       3, 1, 0, 4, 3, 0, 7, 1, 8, 5, 3, 0, 1, 8, 1, 8, 0, 2, 4, 7, 8, 8,
       4, 3, 0, 6, 4, 1, 4, 1])), (array([[22.16577861, 70.20228452],
       [26.6124459 , 43.91604964],
       [19.74208872, 64.1000882 ],
       [39.78587939, 19.72783208],
       [15.57193046, 70.58516939],
       [60.58250295, 27.97262624],
       [25.38793378, 73.72176884],
       [ 9.92511389, 25.73098987],
       [55.37159269, 32.68544864],
       [31.42335817, 38.75662838]]), array([9, 9, 4, 9, 8, 7, 9, 3, 0, 3, 3, 1, 4, 3, 0, 7, 3, 8, 3, 0, 7, 1,
       4, 5, 9, 7, 3, 3, 7, 5, 5, 9, 0, 7, 3, 9, 5, 9, 5, 2, 0, 3, 1, 7,
       3, 1, 9, 9, 6, 5, 5, 4, 1, 6, 9, 7, 9, 7, 9, 9, 7, 7, 8, 7, 5, 5,
       1, 3, 2, 5, 0, 3, 9, 9, 7, 9, 0, 7, 9, 2, 8, 9, 5, 6, 7, 3, 7, 1,
       5, 4, 9, 7, 8, 9, 9, 0, 7, 6, 3, 5, 9, 7, 5, 3, 3, 9, 7, 3, 5, 3,
       7, 7, 3, 4, 3, 7, 7, 6, 7, 7, 3, 3, 1, 0, 9, 0, 4, 8, 4, 7, 9, 7,
       5, 4, 5, 7, 1, 3, 1, 8, 1, 8, 6, 5, 4, 0, 8, 7, 7, 4, 5, 7, 8, 3,
       6, 2, 2, 5, 5, 3, 7, 7, 9, 3, 3, 5, 3, 7, 6, 4, 2, 6, 9, 3, 5, 0,
       7, 5, 8, 7, 3, 3, 7, 0, 7, 4, 9, 3, 6, 2, 5, 3, 8, 9, 3, 3, 5, 8,
       7, 9, 5, 3, 3, 9, 5, 4, 5, 7, 9, 5, 9, 7, 9, 7, 7, 9, 1, 7, 2, 1,
       4, 3, 3, 7, 4, 3, 5, 3, 5, 2, 4, 3, 3, 5, 3, 5, 3, 1, 7, 5, 5, 5,
       7, 2, 3, 9, 7, 3, 7, 3]))]
[     0.       75832.02223 123143.46797 134769.30424 150059.18397
 150520.77443 150997.04159 151765.54636 152083.78065 152200.29395]Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:3-optimum.pyHelp×Students who are done with "3. Optimize k"Review your work×Correction of "3. Optimize k"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as np,kmeans = __import__('1-kmeans').kmeans,andvariance = __import__('2-variance').varianceMaximum of two loops allowedCorrect output: NormalCorrect output: high dimensionalXCorrect output:kmingreater than1Correct output: invalidXCorrect output: invalidkminCorrect output: invalidkmaxCorrect output:kmax>=kminCorrect output: invaliditerationspycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×3. Optimize kCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---10/10pts

4. Initialize GMMmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef initialize(X, k):that initializes variables for a Gaussian Mixture Model:Xis anumpy.ndarrayof shape(n, d)containing the data setkis a positive integer containing the number of clustersYou are not allowed to use any loopsReturns:pi, m, S, orNone, None, Noneon failurepiis anumpy.ndarrayof shape(k,)containing the priors for each cluster, initialized evenlymis anumpy.ndarrayof shape(k, d)containing the centroid means for each cluster, initialized with K-meansSis anumpy.ndarrayof shape(k, d, d)containing the covariance matrices for each cluster, initialized as identity matricesYou should usekmeans = __import__('1-kmeans').kmeansalexa@ubuntu-xenial:0x01-clustering$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    print(pi)
    print(m)
    print(S)
alexa@ubuntu-xenial:0x01-clustering$ ./4-main.py
[0.25 0.25 0.25 0.25]
[[54.73711515 31.81393242]
 [16.84012557 31.20248225]
 [21.43215816 65.50449077]
 [32.3301925  41.80664127]]
[[[1. 0.]
  [0. 1.]]

 [[1. 0.]
  [0. 1.]]

 [[1. 0.]
  [0. 1.]]

 [[1. 0.]
  [0. 1.]]]
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:4-initialize.pyHelp×Students who are done with "4. Initialize GMM"Review your work×Correction of "4. Initialize GMM"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3No loops allowedNot allowed to import anything exceptimport numpy as npandkmeans = __import__('1-kmeans').kmeansCorrect output: NormalCorrect output: High dimensionalXCorrect output: invalidXCorrect output: invalidkpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×4. Initialize GMMCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---6/6pts

5. PDFmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef pdf(X, m, S):that calculates the probability density function of a Gaussian distribution:Xis anumpy.ndarrayof shape(n, d)containing the data points whose PDF should be evaluatedmis anumpy.ndarrayof shape(d,)containing the mean of the distributionSis anumpy.ndarrayof shape(d, d)containing the covariance of the distributionYou are not allowed to use any loopsYou are not allowed to use the functionnumpy.diagor the methodnumpy.ndarray.diagonalReturns:P, orNoneon failurePis anumpy.ndarrayof shape(n,)containing the PDF values for each data pointAll values inPshould have a minimum value of1e-300alexa@ubuntu-xenial:0x01-clustering$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
pdf = __import__('5-pdf').pdf

if __name__ == '__main__':
    np.random.seed(0)
    m = np.array([12, 30, 10])
    S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    X = np.random.multivariate_normal(m, S, 10000)
    P = pdf(X, m, S)
    print(P)
alexa@ubuntu-xenial:0x01-clustering$ ./5-main.py
[3.47450910e-05 2.53649178e-06 1.80348301e-04 ... 1.24604061e-04
 1.86345129e-04 2.59397003e-05]
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:5-pdf.pyHelp×Students who are done with "5. PDF"Review your work×Correction of "5. PDF"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as npandkmeans = __import__('1-kmeans').kmeansNo loops allowedCorrect output: NormalCorrect output: High dimensionalXCorrect output: invalidXCorrect output: invalidmCorrect output: invalidSCorrect output: min value ofXis 1e-300pycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×5. PDFCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---8/8pts

6. ExpectationmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef expectation(X, pi, m, S):that calculates the expectation step in the EM algorithm for a GMM:Xis anumpy.ndarrayof shape(n, d)containing the data setpiis anumpy.ndarrayof shape(k,)containing the priors for each clustermis anumpy.ndarrayof shape(k, d)containing the centroid means for each clusterSis anumpy.ndarrayof shape(k, d, d)containing the covariance matrices for each clusterYou may use at most 1 loopReturns:g, l, orNone, Noneon failuregis anumpy.ndarrayof shape(k, n)containing the posterior probabilities for each data point in each clusterlis the total log likelihoodYou should usepdf = __import__('5-pdf').pdfalexa@ubuntu-xenial:0x01-clustering$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, l = expectation(X, pi, m, S)
    print(g)
    print(np.sum(g, axis=0))
    print(l)
alexa@ubuntu-xenial:0x01-clustering$ ./6-main.py
[[1.98542668e-055 1.00000000e+000 1.56526421e-185 ... 1.00000000e+000
  3.70567311e-236 1.91892348e-012]
 [6.97883333e-085 2.28658376e-279 9.28518983e-065 ... 8.12227631e-287
  1.53690661e-032 3.17417182e-181]
 [9.79811365e-234 2.28658376e-279 2.35073465e-095 ... 1.65904890e-298
  9.62514613e-068 5.67072057e-183]
 [1.00000000e+000 7.21133039e-186 1.00000000e+000 ... 2.42138447e-125
  1.00000000e+000 1.00000000e+000]]
[1. 1. 1. ... 1. 1. 1.]
-652797.7866541843
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:6-expectation.pyHelp×Students who are done with "6. Expectation"Review your work×Correction of "6. Expectation"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as npandpdf = __import__('5-pdf').pdfMaximum of one loop allowedCorrect output: NormalCorrect output: High dimensionalXCorrect output: invalidXCorrect output: invalidpiCorrect output: invalidmCorrect output: invalidSpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×6. ExpectationCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---8/8pts

7. MaximizationmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef maximization(X, g):that calculates the maximization step in the EM algorithm for a GMM:Xis anumpy.ndarrayof shape(n, d)containing the data setgis anumpy.ndarrayof shape(k, n)containing the posterior probabilities for each data point in each clusterYou may use at most 1 loopReturns:pi, m, S, orNone, None, Noneon failurepiis anumpy.ndarrayof shape(k,)containing the updated priors for each clustermis anumpy.ndarrayof shape(k, d)containing the updated centroid means for each clusterSis anumpy.ndarrayof shape(k, d, d)containing the updated covariance matrices for each clusteralexa@ubuntu-xenial:0x01-clustering$ cat 7-main.py
#!/usr/bin/env python3

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, _ = expectation(X, pi, m, S)
    pi, m, S = maximization(X, g)
    print(pi)
    print(m)
    print(S)
alexa@ubuntu-xenial:0x01-clustering$ ./7-main.py
[0.10104901 0.24748822 0.1193333  0.53212947]
[[54.7440558  31.80888393]
 [16.84099873 31.20560148]
 [21.42588061 65.51441875]
 [32.33208369 41.80830251]]
[[[64.05063663 -2.13941814]
  [-2.13941814 41.90354928]]

 [[72.72404579  9.96322554]
  [ 9.96322554 53.05035303]]

 [[46.20933259  1.08979413]
  [ 1.08979413 66.9841323 ]]

 [[35.04054823 -0.94790014]
  [-0.94790014 45.14948772]]]
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:7-maximization.pyHelp×Students who are done with "7. Maximization"Review your work×Correction of "7. Maximization"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Maximum of one loop allowedNot allowed to import anything exceptimport numpy as npandkmeans = __import__('1-kmeans').kmeansCorrect output: NormalCorrect output: High dimensionalXCorrect output: invalidXCorrect output: invalidgpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×7. MaximizationCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---6/6pts

8. EMmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):that performs the expectation maximization for a GMM:Xis anumpy.ndarrayof shape(n, d)containing the data setkis a positive integer containing the number of clustersiterationsis a positive integer containing the maximum number of iterations for the algorithmtolis a non-negative float containing tolerance of the log likelihood, used to determine early stopping i.e. if the difference is less than or equal totolyou should stop the algorithmverboseis a boolean that determines if you should print information about the algorithmIfTrue, printLog Likelihood after {i} iterations: {l}every 10 iterations and after the last iteration{i}is the number of iterations of the EM algorithm{l}is the log likelihood, rounded to 5 decimal placesYou should use:initialize = __import__('4-initialize').initializeexpectation = __import__('6-expectation').expectationmaximization = __import__('7-maximization').maximizationYou may use at most 1 loopReturns:pi, m, S, g, l, orNone, None, None, None, Noneon failurepiis anumpy.ndarrayof shape(k,)containing the priors for each clustermis anumpy.ndarrayof shape(k, d)containing the centroid means for each clusterSis anumpy.ndarrayof shape(k, d, d)containing the covariance matrices for each clustergis anumpy.ndarrayof shape(k, n)containing the probabilities for each data point in each clusterlis the log likelihood of the modelalexa@ubuntu-xenial:0x01-clustering$ cat 8-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
    print(X.shape[0] * pi)
    print(m)
    print(S)
    print(l)
alexa@ubuntu-xenial:0x01-clustering$ ./8-main.py
Log Likelihood after 0 iterations: -652797.78665
Log Likelihood after 10 iterations: -94855.45662
Log Likelihood after 20 iterations: -94714.52057
Log Likelihood after 30 iterations: -94590.87362
Log Likelihood after 40 iterations: -94440.40559
Log Likelihood after 50 iterations: -94439.93891
Log Likelihood after 52 iterations: -94439.93889[ 761.03239903  747.62391034 1005.60275934 9985.74093129]
[[60.18888335 30.19707607]
 [ 5.05794926 24.92588821]
 [20.03438453 69.84721009]
 [29.89607379 40.12519148]]
[[[16.85183426  0.2547388 ]
  [ 0.2547388  16.49432111]]

 [[15.19555672  9.62661086]
  [ 9.62661086 15.47295413]]

 [[35.58332494 11.08419454]
  [11.08419454 33.09463207]]

 [[74.52083678  5.20755533]
  [ 5.20755533 73.87299705]]]
-94439.93889004056
alexa@ubuntu-xenial:0x01-clustering$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:8-EM.pyHelp×Students who are done with "8. EM"Review your work×Correction of "8. EM"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as np,initialize = __import__('4-initialize').initialize,expectation = __import__('6-expectation').expectation, andmaximization = __import__('7-maximization').maximizationMaximum of one loop allowedCorrect output: NormalCorrect output: high dimensionalXCorrect output:tolis1e-6Correct output:iterationsstop beforetolCorrect output:verboseisFalseCorrect output: invalidXCorrect output: invalidkCorrect output: invaliditerationsCorrect output: invalidtolCorrect output: invalidverbosepycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×8. EMCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---12/12pts

9. BICmandatoryScore:92.86%(Checks completed: 92.86%)Write a functiondef BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):that finds the best number of clusters for a GMM using the Bayesian Information Criterion:Xis anumpy.ndarrayof shape(n, d)containing the data setkminis a positive integer containing the minimum number of clusters to check for (inclusive)kmaxis a positive integer containing the maximum number of clusters to check for (inclusive)IfkmaxisNone,kmaxshould be set to the maximum number of clusters possibleiterationsis a positive integer containing the maximum number of iterations for the EM algorithmtolis a non-negative float containing the tolerance for the EM algorithmverboseis a boolean that determines if the EM algorithm should print information to the standard outputYou should useexpectation_maximization = __import__('8-EM').expectation_maximizationYou may use at most 1 loopReturns:best_k, best_result, l, b, orNone, None, None, Noneon failurebest_kis the best value for k based on its BICbest_resultis tuple containingpi, m, Spiis anumpy.ndarrayof shape(k,)containing the cluster priors for the best number of clustersmis anumpy.ndarrayof shape(k, d)containing the centroid means for the best number of clustersSis anumpy.ndarrayof shape(k, d, d)containing the covariance matrices for the best number of clusterslis anumpy.ndarrayof shape(kmax - kmin + 1)containing the log likelihood for each cluster size testedbis anumpy.ndarrayof shape(kmax - kmin + 1)containing the BIC value for each cluster size testedUse:BIC = p * ln(n) - 2 * lpis the number of parameters required for the modelnis the number of data points used to create the modellis the log likelihood of the modelalexa@ubuntu-xenial:0x01-clustering$ cat 9-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
    print(best_k)
    print(best_result)
    print(l)
    print(b)
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./9-main.py
4
(array([0.79885962, 0.08044842, 0.06088258, 0.05980938]), array([[29.89606417, 40.12518027],
       [20.0343883 , 69.84718588],
       [60.18888407, 30.19707372],
       [ 5.05788987, 24.92583792]]), array([[[74.52101284,  5.20770764],
        [ 5.20770764, 73.8729309 ]],

       [[35.58334497, 11.08416742],
        [11.08416742, 33.09483747]],

       [[16.85183256,  0.25475122],
        [ 0.25475122, 16.4943092 ]],

       [[15.19520213,  9.62633552],
        [ 9.62633552, 15.47268905]]]))
[-98801.40298366 -96729.95558846 -95798.40406023 -94439.93888882
 -94435.87750008 -94428.62217176 -94426.71159745 -94425.5860871
 -94421.41864281 -94416.43390835]
[197649.97338694 193563.67950008 191757.17734716 189096.84790787
 189145.32603394 189187.41628084 189240.19603576 189294.54591859
 189342.81193356 189389.44336818]Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:9-BIC.pyHelp×Students who are done with "9. BIC"Review your work×Correction of "9. BIC"Some checks are failing. Make sure you fix them before starting a new reviewYou got this!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport numpy as npandexpectation_maximization = __import__('8-EM').expectation_maximizationMaximum of one loop allowedCorrect output: NormalCorrect output: high dimensionalXCorrect output:tolis1e-6Correct output:iterationsstops beforetolCorrect output:verboseisTrue[copy_files] Filed copied: 4-main.py
[compare] Command to run:
./4-main.py
su student_jail -c 'timeout 300 bash -c ./4-main.py'
[compare] Return code: 0
[compare] Student stdout:
Log Likelihood after 0 iterations: -1990052.02613
Log Likelihood after 2 iterations: -98801.40298
Log Likelihood after 0 iterations: -1458487.49502
Log Likelihood after 10 iterations: -96792.91725
Log Likelihood after 12 iterations: -96782.36971
Log Likelihood after 0 iterations: -909558.88536
Log Likelihood after 10 iterations: -95939.45733
Log Likelihood after 12 iterations: -95895.42976
Log Likelihood after 0 iterations: -652799.84729
Log Likelihood after 10 iterations: -94853.57712
Log Likelihood after 12 iterations: -94805.34151
Log Likelihood after 0 iterations: -508386.90105
Log Likelihood after 10 iterations: -94567.77709
Log Likelihood after 12 iterations: -94519.71482
Log Likelihood after 0 iterations: -398063.68216
Log Likelihood after 10 iterations: -94483.50744
Log Likelihood after 12 iterations: -94469.13514
Log Likelihood after 0 iterations: -346470.73290
Log Likelihood after 10 iterations: -94474.70092
Log Likelihood after 12 iterations: -94462.34292
Log Likelihood after 0 iterations: -313456.83650
Log Likelihood after 10 iterations: -94468.92162
Log Likelihood after 12 iterations: -94458.80640
Log Likelihood after 0 iterations: -282512.76159
Log Likelihood after 10 iterations: -94474.91758
Log Likelihood after 12 iterations: -94463.51956
Log Likelihood after 0 iterations: -262710.13701
Log Likelihood after 10 iterations: -94481.44472
Log Likelihood after 12 iterations: -94468.05982
6
(array([0.27508932, 0.06149804, 0.08361659, 0.24816616, 0.2679412 ,
       0.06368869]), array([[29.43833013, 33.65231436],
       [60.11893477, 30.20852505],
       [20.04418883, 69.47274931],
       [24.50070318, 42.49830097],
       [35.69726168, 44.50500557],
       [ 5.4177726 , 25.23462058]]), array([[[53.2150377 ,  3.51985925],
        [ 3.51985925, 47.20680701]],
       [[17.33883115,  0.29166467],
        [ 0.29166467, 16.74253093]],
       [[36.25345477, 11.26915415],
        [11.26915415, 36.26267798]],
       [[49.53304529,  3.68513373],
        [ 3.68513373, 44.38804834]],
       [[52.43576011, -8.45619192],
        [-8.45619192, 56.3483049 ]],
       [[17.14910231, 11.03171459],
        [11.03171459, 16.97284825]]]))
[-98801.40298 -96782.36971 -95895.42976 -94805.34151 -94519.71482
 -94469.13514 -94462.34292 -94458.8064  -94463.51956 -94468.05982]
[197649.97338694 193668.50775153 191951.22874421 189827.65315696
 189313.00067259 189268.4422109  189311.45868923 189360.98654538
 189427.01377013 189492.69518568]
[compare] Student stdout length: 2467
[compare] Student stderr:
[compare] Student stderr length: 0
[compare] Desired stdout:
Log Likelihood after 0 iterations: -1990052.02613
Log Likelihood after 2 iterations: -98801.40298
Log Likelihood after 0 iterations: -1458487.49502
Log Likelihood after 10 iterations: -96792.91725
Log Likelihood after 12 iterations: -96782.36971
Log Likelihood after 0 iterations: -909558.88536
Log Likelihood after 10 iterations: -95939.45733
Log Likelihood after 12 iterations: -95895.42976
Log Likelihood after 0 iterations: -652799.84729
Log Likelihood after 10 iterations: -94853.57712
Log Likelihood after 12 iterations: -94805.34151
Log Likelihood after 0 iterations: -508386.90105
Log Likelihood after 10 iterations: -94567.77709
Log Likelihood after 12 iterations: -94519.71482
Log Likelihood after 0 iterations: -398063.68216
Log Likelihood after 10 iterations: -94483.50744
Log Likelihood after 12 iterations: -94469.13514
Log Likelihood after 0 iterations: -346470.7329
Log Likelihood after 10 iterations: -94474.70092
Log Likelihood after 12 iterations: -94462.34292
Log Likelihood after 0 iterations: -313456.8365
Log Likelihood after 10 iterations: -94468.92162
Log Likelihood after 12 iterations: -94458.8064
Log Likelihood after 0 iterations: -282512.76159
Log Likelihood after 10 iterations: -94474.91758
Log Likelihood after 12 iterations: -94463.51956
Log Likelihood after 0 iterations: -262710.13701
Log Likelihood after 10 iterations: -94481.44472
Log Likelihood after 12 iterations: -94468.05982
6
(array([0.27508932, 0.06149804, 0.08361659, 0.24816616, 0.2679412 ,
       0.06368869]), array([[29.43833013, 33.65231436],
       [60.11893477, 30.20852505],
       [20.04418883, 69.47274931],
       [24.50070318, 42.49830097],
       [35.69726168, 44.50500557],
       [ 5.4177726 , 25.23462058]]), array([[[53.2150377 ,  3.51985925],
        [ 3.51985925, 47.20680701]],
       [[17.33883115,  0.29166467],
        [ 0.29166467, 16.74253093]],
       [[36.25345477, 11.26915415],
        [11.26915415, 36.26267798]],
       [[49.53304529,  3.68513373],
        [ 3.68513373, 44.38804834]],
       [[52.43576011, -8.45619192],
        [-8.45619192, 56.3483049 ]],
       [[17.14910231, 11.03171459],
        [11.03171459, 16.97284825]]]))
[-98801.40298 -96782.36971 -95895.42976 -94805.34151 -94519.71482
 -94469.13514 -94462.34292 -94458.8064  -94463.51956 -94468.05982]
[197649.97338694 193668.50775153 191951.22874421 189827.65315696
 189313.00067259 189268.4422109  189311.45868923 189360.98654538
 189427.01377013 189492.69518568]
[compare] Desired stdout length: 2464Correct output: invalidXCorrect output: invalidkminCorrect output: invalidkmaxCorrect output:kmax>=kminCorrect output: invaliditerationsCorrect output: invalidtolCorrect output: invalidverbosepycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×9. BICCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---13/14pts

10. Hello, sklearn!mandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef kmeans(X, k):that performs K-means on a dataset:Xis anumpy.ndarrayof shape(n, d)containing the datasetkis the number of clustersThe only import you are allowed to use isimport sklearn.clusterReturns:C, clssCis anumpy.ndarrayof shape(k, d)containing the centroid means for each clusterclssis anumpy.ndarrayof shape(n,)containing the index of the cluster inCthat each data point belongs toalexa@ubuntu-xenial:0x01-clustering$ cat 10-main.py
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('10-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./10-main.py
[[30.06722465 40.41123947]
 [59.22766628 29.19796006]
 [ 9.92511389 25.73098987]
 [20.0835633  69.81592298]
 [39.62770705 19.89843487]]Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:10-kmeans.pyHelp×Students who are done with "10. Hello, sklearn!"Review your work×Correction of "10. Hello, sklearn!"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport sklearn.clusterCorrect output: NormalCorrect output: High dimensionalXpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×10. Hello, sklearn!Commit used:User:---URL:Click hereID:---Author:---Subject:---Date:---4/4pts

11. GMMmandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef gmm(X, k):that calculates a GMM from a dataset:Xis anumpy.ndarrayof shape(n, d)containing the datasetkis the number of clustersThe only import you are allowed to use isimport sklearn.mixtureReturns:pi, m, S, clss, bicpiis anumpy.ndarrayof shape(k,)containing the cluster priorsmis anumpy.ndarrayof shape(k, d)containing the centroid meansSis anumpy.ndarrayof shape(k, d, d)containing the covariance matricesclssis anumpy.ndarrayof shape(n,)containing the cluster indices for each data pointbicis anumpy.ndarrayof shape(kmax - kmin + 1)containing the BIC value for each cluster size testedalexa@ubuntu-xenial:0x01-clustering$ cat 11-main.py
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
gmm = __import__('11-gmm').gmm

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    pi, m, S, clss, bic = gmm(X, 4)
    print(pi)
    print(m)
    print(S)
    print(bic)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, marker='*', c=list(range(4)))
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./11-main.py
[0.68235847 0.17835524 0.06079717 0.07848911]
[[30.6032058  40.76046968]
 [18.77413861 32.82638046]
 [60.2071779  30.24071437]
 [20.00542639 70.02497115]]
[[[ 6.96776628e+01 -3.66204575e+00]
  [-3.66204575e+00  7.82087329e+01]]

 [[ 1.54935691e+02  8.17658780e+01]
  [ 8.17658780e+01  6.56306521e+01]]

 [[ 1.66980918e+01  1.00778176e-02]
  [ 1.00778176e-02  1.66498476e+01]]

 [[ 3.55102735e+01  1.13250004e+01]
  [ 1.13250004e+01  3.21335458e+01]]]
189794.11897876553Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:11-gmm.pyHelp×Students who are done with "11. GMM"Review your work×Correction of "11. GMM"Congratulations!All tests passed successfully!You are ready for your next mission!Start a new testCloseResult:File existsFirst line contains#!/usr/bin/env python3Not allowed to import anything exceptimport sklearn.mixtureCorrect output: NormalCorrect output: High dimensionalXpycodestyle validationEverything is documentedRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedQA Review×11. GMMCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---4/4pts

12. AgglomerativemandatoryScore:100.00%(Checks completed: 100.00%)Write a functiondef agglomerative(X, dist):that performs agglomerative clustering on a dataset:Xis anumpy.ndarrayof shape(n, d)containing the datasetdistis the maximum cophenetic distance for all clustersPerforms agglomerative clustering with Ward linkageDisplays the dendrogram with each cluster displayed in a different colorThe only imports you are allowed to use are:import scipy.cluster.hierarchyimport matplotlib.pyplot as pltReturns:clss, anumpy.ndarrayof shape(n,)containing the cluster indices for each data pointalexa@ubuntu-xenial:0x01-clustering$ cat 12-main.py
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
agglomerative = __import__('12-agglomerative').agglomerative

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=100)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=100)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    clss = agglomerative(X, 100)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./12-main.pyRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/clusteringFile:12-agglomerative.pyHelp×Students who are done with "12. Agglomerative"QA Review×12. AgglomerativeCommit used:User:---URL:Click hereID:---Author:---Subject:---Date:---6/6pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Clustering.md`
