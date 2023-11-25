from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import numpy as np
from jmetal.core.problem import FloatProblem, S
from jmetal.problem.machine_learning.ensemble_ml_problem import EnsembleMLProblem
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline


class WrapperMLProblem(EnsembleMLProblem):

    def __init__(self, X_train, X_test, y_train, y_test, seed):
        super().__init__(X_train, X_test, y_train, y_test, seed)
        self.number_of_variables = 9
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Log-Loss']

        self.seed = seed

        self.rf = None
        self.svm = None
        self.lg = None
        self.knn = None
        self.lda = None

        self.voting = None
        self.sfs = None

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        np.random.seed(self.seed)

    def update_classifiers_parameters(self, solution: S):
        self.rf = RandomForestClassifier(n_estimators=abs(int(solution.variables[0])),
                                         ccp_alpha=abs(solution.variables[1]),
                                         random_state=self.seed,
                                         bootstrap=True, max_features= 'log2')
        self.svm = SVC(kernel='linear',
                       gamma='scale',
                       C=10**int(solution.variables[2]),
                       probability=True,
                       random_state=self.seed)

        self.lg = LogisticRegression(solver='saga',
                                     max_iter=abs(int(solution.variables[3])),
                                     C=10**int(solution.variables[4]),
                                     tol = 10**solution.variables[5],
                                     random_state=self.seed)

        self.knn = KNeighborsClassifier(n_neighbors=abs(int(solution.variables[6])), metric='chebyshev', weights = 'distance')

        self.lda = LinearDiscriminantAnalysis(solver='svd',
                                              tol=10**solution.variables[7])

        self.voting = VotingClassifier(
            estimators=[('SVM', self.svm),
                        ('Random Forests', self.rf),
                        ('LogReg', self.lg),
                        ('KNN', self.knn),
                        ('LDA', self.lda)],
            voting='soft')

        self.sfs = SFS(self.voting,
                       k_features=int(solution.variables[8]),
                       forward=True,
                       floating=False,
                       scoring='neg_log_loss')
        self.pipe = Pipeline([
            ('feature_selection', self.sfs),
            ('classification', self.voting)
        ])

    def evaluate(self, solution: S) -> S:
        self.update_classifiers_parameters(solution)
        pipeFit = self.pipe.fit(self.X_train, self.y_train)
        predictions = pipeFit.predict_proba(self.X_test)
        score = log_loss(self.y_test, predictions)
        # wrapperFit = self.sfs.fit(self.X_train, self.y_train)
        #
        # X_train_sfs = wrapperFit.transform(self.X_train)
        # X_test_sfs = wrapperFit.transform(self.X_test)
        #
        # # Fit the estimator using the new feature subset
        # # and make a prediction on the test data
        # classifierFit = self.voting.fit(X_train_sfs, self.y_train)
        # predictions = classifierFit.predict_proba(X_test_sfs)
        # score = log_loss(self.y_test, predictions)

        solution.objectives[0] = score

        return solution

    def get_name(self) -> str:
        return 'WrapperEnsemble-ML'
