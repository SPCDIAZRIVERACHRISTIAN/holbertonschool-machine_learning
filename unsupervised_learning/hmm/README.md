
## Hidden Markov Models

### Description
0. Markov ChainmandatoryWrite the functiondef markov_chain(P, s, t=1):that determines the probability of a markov chain being in a particular state after a specified number of iterations:Pis a square 2Dnumpy.ndarrayof shape(n, n)representing the transition matrixP[i, j]is the probability of transitioning from stateito statejnis the number of states in the markov chainsis anumpy.ndarrayof shape(1, n)representing the probability of starting in each statetis the number of iterations that the markov chain has been throughReturns: anumpy.ndarrayof shape(1, n)representing the probability of being in a specific state aftertiterations, orNoneon failurealexa@ubuntu-xenial:0x02-hmm$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
markov_chain = __import__('0-markov_chain').markov_chain

if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
alexa@ubuntu-xenial:0x02-hmm$ ./0-main.py
[[0.2494929  0.26335362 0.23394185 0.25321163]]
alexa@ubuntu-xenial:0x02-hmm$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:0-markov_chain.pyHelp×Students who are done with "0. Markov Chain"Review your work×Correction of "0. Markov Chain"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/3pts

1. Regular ChainsmandatoryWrite the functiondef regular(P):that determines the steady state probabilities of a regular markov chain:Pis a is a square 2Dnumpy.ndarrayof shape(n, n)representing the transition matrixP[i, j]is the probability of transitioning from stateito statejnis the number of states in the markov chainReturns: anumpy.ndarrayof shape(1, n)containing the steady state probabilities, orNoneon failurealexa@ubuntu-xenial:0x02-hmm$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
regular = __import__('1-regular').regular

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[0.8, 0.2, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0, 0.5, 0.2, 0.3],
                [0, 0, 0.3, 0.5, .2],
                [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0.25, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0.1, 0.5, 0.2, 0.2],
                [0, 0.1, 0.2, 0.5, .2],
                [0, 0.1, 0.2, 0.2, 0.5]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
    print(regular(d))
    print(regular(e))
alexa@ubuntu-xenial:0x02-hmm$ ./1-main.py
None
[[0.42857143 0.57142857]]
[[0.2494929  0.26335362 0.23394185 0.25321163]]
None
None
alexa@ubuntu-xenial:0x02-hmm$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:1-regular.pyHelp×Students who are done with "1. Regular Chains"Review your work×Correction of "1. Regular Chains"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

2. Absorbing ChainsmandatoryWrite the functiondef absorbing(P):that determines if a markov chain is absorbing:P is a is a square 2Dnumpy.ndarrayof shape(n, n)representing the standard transition matrixP[i, j]is the probability of transitioning from stateito statejnis the number of states in the markov chainReturns:Trueif it is absorbing, orFalseon failurealexa@ubuntu-xenial:0x02-hmm$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
absorbing = __import__('2-absorbing').absorbing

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0, 0.5, 0.2, 0.3],
                  [0, 0, 0.3, 0.5, .2],
                  [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0.1, 0.5, 0.2, 0.2],
                  [0, 0.1, 0.2, 0.5, .2],
                  [0, 0.1, 0.2, 0.2, 0.5]])
    f = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0.5, 0.5],
                  [0, 0.5, 0.5, 0]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
    print(absorbing(d))
    print(absorbing(e))
    print(absorbing(f))
alexa@ubuntu-xenial:0x02-hmm$ ./2-main.py
True
False
False
False
True
True
alexa@ubuntu-xenial:0x02-hmm$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:2-absorbing.pyHelp×Students who are done with "2. Absorbing Chains"Review your work×Correction of "2. Absorbing Chains"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/8pts

3. The Forward AlgorithmmandatoryWrite the functiondef forward(Observation, Emission, Transition, Initial):that performs the forward algorithm for a hidden markov model:Observationis anumpy.ndarrayof shape(T,)that contains the index of the observationTis the number of observationsEmissionis anumpy.ndarrayof shape(N, M)containing the emission probability of a specific observation given a hidden stateEmission[i, j]is the probability of observingjgiven the hidden stateiNis the number of hidden statesMis the number of all possible observationsTransitionis a 2Dnumpy.ndarrayof shape(N, N)containing the transition probabilitiesTransition[i, j]is the probability of transitioning from the hidden stateitojInitialanumpy.ndarrayof shape(N, 1)containing the probability of starting in a particular hidden stateReturns:P, F, orNone, Noneon failurePis the likelihood of the observations given the modelFis anumpy.ndarrayof shape(N, T)containing  the forward path probabilitiesF[i, j]is the probability of being in hidden stateiat timejgiven the previous observationsalexa@ubuntu-xenial:0x02-hmm$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np
forward = __import__('3-forward').forward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, F = forward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(F)
alexa@ubuntu-xenial:0x02-hmm$ ./3-main.py
1.7080966131859584e-214
[[0.00000000e+000 0.00000000e+000 2.98125000e-004 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.00000000e-002 0.00000000e+000 3.18000000e-003 ... 0.00000000e+000
  0.00000000e+000 0.00000000e+000]
 [2.50000000e-001 3.31250000e-002 0.00000000e+000 ... 2.13885975e-214
  1.17844112e-214 0.00000000e+000]
 [1.00000000e-002 4.69000000e-002 0.00000000e+000 ... 2.41642482e-213
  1.27375484e-213 9.57568349e-215]
 [0.00000000e+000 8.00000000e-004 0.00000000e+000 ... 1.96973759e-214
  9.65573676e-215 7.50528264e-215]]
alexa@ubuntu-xenial:0x02-hmm$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:3-forward.pyHelp×Students who are done with "3. The Forward Algorithm"Review your work×Correction of "3. The Forward Algorithm"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/3pts

4. The Viretbi AlgorithmmandatoryWrite the functiondef viterbi(Observation, Emission, Transition, Initial):that calculates the most likely sequence of hidden states for a hidden markov model:Observationis anumpy.ndarrayof shape(T,)that contains the index of the observationTis the number of observationsEmissionis anumpy.ndarrayof shape(N, M)containing the emission probability of a specific observation given a hidden stateEmission[i, j]is the probability of observingjgiven the hidden stateiNis the number of hidden statesMis the number of all possible observationsTransitionis a 2Dnumpy.ndarrayof shape(N, N)containing the transition probabilitiesTransition[i, j]is the probability of transitioning from the hidden stateitojInitialanumpy.ndarrayof shape(N, 1)containing the probability of starting in a particular hidden stateReturns:path, P, orNone, Noneon failurepathis the a list of lengthTcontaining the most likely sequence of hidden statesPis the probability of obtaining thepathsequencealexa@ubuntu-xenial:0x02-hmm$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
viterbi = __import__('4-viterbi').viterbi

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    path, P = viterbi(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(path)
alexa@ubuntu-xenial:0x02-hmm$ ./4-main.py
4.701733355108224e-252
[2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 3, 3, 3, 2, 1, 2, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 4, 4, 3, 3, 2, 2, 3, 3, 3, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0, 0, 1, 2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3]
alexa@ubuntu-xenial:0x02-hmm$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:4-viterbi.pyHelp×Students who are done with "4. The Viretbi Algorithm"Review your work×Correction of "4. The Viretbi Algorithm"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/3pts

5. The Backward AlgorithmmandatoryWrite the functiondef backward(Observation, Emission, Transition, Initial):that performs the backward algorithm for a hidden markov model:Observationis anumpy.ndarrayof shape(T,)that contains the index of the observationTis the number of observationsEmissionis anumpy.ndarrayof shape(N, M)containing the emission probability of a specific observation given a hidden stateEmission[i, j]is the probability of observingjgiven the hidden stateiNis the number of hidden statesMis the number of all possible observationsTransitionis a 2Dnumpy.ndarrayof shape(N, N)containing the transition probabilitiesTransition[i, j]is the probability of transitioning from the hidden stateitojInitialanumpy.ndarrayof shape(N, 1)containing the probability of starting in a particular hidden stateReturns:P, B, orNone, Noneon failurePis the likelihood of the observations given the modelBis anumpy.ndarrayof shape(N, T)containing  the backward path probabilitiesB[i, j]is the probability of generating the future observations from hidden stateiat timejalexa@ubuntu-xenial:0x02-hmm$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
backward = __import__('5-backward').backward

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    P, B = backward(Observations, Emission, Transition, Initial.reshape((-1, 1)))
    print(P)
    print(B)
alexa@ubuntu-xenial:0x02-hmm$ ./5-main.py
1.7080966131859631e-214
[[1.28912952e-215 6.12087935e-212 1.00555701e-211 ... 6.75000000e-005
  0.00000000e+000 1.00000000e+000]
 [3.86738856e-214 2.69573528e-212 4.42866330e-212 ... 2.02500000e-003
  0.00000000e+000 1.00000000e+000]
 [6.44564760e-214 5.15651808e-213 8.47145100e-213 ... 2.31330000e-002
  2.70000000e-002 1.00000000e+000]
 [1.93369428e-214 0.00000000e+000 0.00000000e+000 ... 6.39325000e-002
  1.15000000e-001 1.00000000e+000]
 [1.28912952e-215 0.00000000e+000 0.00000000e+000 ... 5.77425000e-002
  2.19000000e-001 1.00000000e+000]]
alexa@ubuntu-xenial:0x02-hmm$Repo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:5-backward.pyHelp×Students who are done with "5. The Backward Algorithm"Review your work×Correction of "5. The Backward Algorithm"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/3pts

6. The Baum-Welch AlgorithmmandatoryWrite the functiondef baum_welch(Observations, Transition, Emission, Initial, iterations=1000):that performs the Baum-Welch algorithm for a hidden markov model:Observationsis anumpy.ndarrayof shape(T,)that contains the index of the observationTis the number of observationsTransitionis anumpy.ndarrayof shape(M, M)that contains the initialized transition probabilitiesMis the number of hidden statesEmissionis anumpy.ndarrayof shape(M, N)that contains the initialized emission probabilitiesNis the number of output statesInitialis anumpy.ndarrayof shape(M, 1)that contains the initialized starting probabilitiesiterationsis the number of times expectation-maximization should be performedReturns: the convergedTransition, Emission, orNone, Noneon failurealexa@ubuntu-xenial:0x02-hmm$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
baum_welch = __import__('6-baum_welch').baum_welch

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00],
                         [0.40, 0.50, 0.10]])
    Transition = np.array([[0.60, 0.4],
                           [0.30, 0.70]])
    Initial = np.array([0.5, 0.5])
    Hidden = [np.random.choice(2, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(3, p=Emission[s]))
    Observations = np.array(Observations)
    T_test = np.ones((2, 2)) / 2
    E_test = np.abs(np.random.randn(2, 3))
    E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))
    T, E = baum_welch(Observations, T_test, E_test, Initial.reshape((-1, 1)))
    print(np.round(T, 2))
    print(np.round(E, 2))
alexa@ubuntu-xenial:0x02-hmm$ ./6-main.py
[[0.81 0.19]
 [0.28 0.72]]
[[0.82 0.18 0.  ]
 [0.26 0.58 0.16]]
alexa@ubuntu-xenial:0x02-hmm$With very little data (only 365 observations), we have been able to get a pretty good estimate of the transition and emission probabilities. We have not used a larger sample size in this example because our implementation does not utilize logarithms to handle values approaching 0 with the increased sequence lengthRepo:GitHub repository:holbertonschool-machine_learningDirectory:unsupervised_learning/hmmFile:6-baum_welch.pyHelp×Students who are done with "6. The Baum-Welch Algorithm"Review your work×Correction of "6. The Baum-Welch Algorithm"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/3pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Hidden_Markov_Models.md`
