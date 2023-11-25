from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from jmetal.core.problem import FloatProblem, S


class EnsembleMLProblem(FloatProblem):

    def __init__(self, X_train, X_test, y_train, y_test):#, seed):
        super().__init__()
        self.number_of_variables = 7
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Log-Loss']

        #self.seed = seed

        self.rf = None
        self.svm = None
        self.lg = None
        self.knn = None
        self.lda = None
        self.voting = None

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #np.random.seed(self.seed)

    def update_classifiers_parameters(self, solution: S):
        self.rf = RandomForestClassifier(n_estimators=abs(int(solution.variables[0])),
                                         ccp_alpha=abs(solution.variables[1]),

                                         bootstrap=True, max_features= 'auto')# random_state=self.seed,
        # self.rf = RandomForestClassifier(n_estimators=abs(int(solution.variables[0])),
        #                                  ccp_alpha=abs(solution.variables[1]),
        #                                  random_state=self.seed,
        #                                  bootstrap=True, max_features= 'log2') #microarray


        self.svm = SVC(
                       C=10**int(solution.variables[2]),
                       probability=True)
                       #,random_state=self.seed)
        # self.svm = SVC(kernel='linear',
        #                gamma='scale',
        #                C=10**int(solution.variables[2]),
        #                probability=True,
        #                random_state=self.seed) #microarray

        #self.lg = LogisticRegression(solver='saga',max_iter=abs(int(solution.variables[3])),C=10**int(solution.variables[4]),tol = 10**solution.variables[5], random_state=self.seed) #microarray
        self.lg = LogisticRegression(solver='newton-cg', max_iter=abs(int(solution.variables[3])),
                                     C=10 ** int(solution.variables[4]), tol=10 ** solution.variables[5])
                                     #random_state=self.seed)
        #self.knn = KNeighborsClassifier(n_neighbors=abs(int(solution.variables[6])), metric='chebyshev', weights = 'distance') #microarray
        #self.knn = KNeighborsClassifier(n_neighbors=abs(int(solution.variables[6])), metric='manhattan',
                                        #weights='distance')
        self.lda = LinearDiscriminantAnalysis(solver='svd',
                                              tol=10**solution.variables[6])

        self.voting = VotingClassifier(
            estimators=[('SVM', self.svm),
                        ('Random Forests', self.rf),
                        ('LogReg', self.lg),
                        #('KNN', self.knn),
                        ('LDA', self.lda)],
            voting='soft')
    def evaluate(self, solution: S) -> S:
        self.update_classifiers_parameters(solution)
        classifierFit = self.voting.fit(self.X_train, self.y_train)
        predictions = classifierFit.predict_proba(self.X_test)
        score = log_loss(self.y_test, predictions)

        solution.objectives[0] = score

        return solution

    def get_name(self) -> str:
        return 'Ensemble'
