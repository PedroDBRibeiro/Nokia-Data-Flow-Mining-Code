from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from jmetal.core.problem import FloatProblem, S
from jmetal.problem.machine_learning.ensemble_ml_problem import EnsembleMLProblem
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline


class PipelineMLProblem_17Dim(EnsembleMLProblem):

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.number_of_variables = 17
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.attributes = {}
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Log-Loss']

        # self.seed = seed

        self.rf = None
        self.svm = None
        self.lg = None
        self.knn = None
        self.lda = None

        self.voting = None
        self.pipe = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # np.random.seed(self.seed)

    def update_classifiers_parameters(self, solution: S):
        self.rf = RandomForestClassifier(n_estimators=abs(int(solution.variables[0])),
                                         ccp_alpha=abs(solution.variables[1]),
                                         bootstrap=True,
                                         max_samples=float(solution.variables[2]),
                                         max_features=float(solution.variables[3]))
        self.svm = SVC(
            C=10 ** int(solution.variables[4]),
            probability=True)

        self.lg = LogisticRegression(solver='saga', penalty='elasticnet',
                                     max_iter=abs(int(solution.variables[5])),
                                     C=10 ** int(solution.variables[6]),
                                     tol=10 ** solution.variables[7],
                                     l1_ratio=solution.variables[8])

        self.knn = KNeighborsClassifier(n_neighbors=abs(int(solution.variables[9])), metric='chebyshev',
                                        weights='distance')

        self.lda = LinearDiscriminantAnalysis(solver='svd',
                                              tol=10 ** solution.variables[10])

        self.voting = VotingClassifier(
            estimators=[('SVM', self.svm),
                        ('Random Forests', self.rf),
                        ('LogReg', self.lg),
                        ('KNN', self.knn),
                        ('LDA', self.lda)],weights=[solution.variables[11],
                                                    solution.variables[12],
                                                    solution.variables[13],
                                                    solution.variables[14],
                                                    solution.variables[15]], voting='soft')

        self.kbest = SelectKBest(k=int(solution.variables[16]))

        self.pipe = Pipeline([
            ('feature_selection', self.kbest),
            ('classification', self.voting)

        ])

    def evaluate(self, solution: S) -> S:
        self.update_classifiers_parameters(solution)
        pipeFit = self.pipe.fit(self.X_train, self.y_train)
        solution.attributes[0] = pipeFit.named_steps['feature_selection'].get_support(indices=True)
        predictions = pipeFit.predict_proba(self.X_test)
        score = log_loss(self.y_test, predictions)

        solution.objectives[0] = score

        return solution

    def get_name(self) -> str:
        return 'BIO-PIPELINE'


class PipelineMLProblem(EnsembleMLProblem):

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.number_of_variables = 9
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.attributes = {}
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Log-Loss']

        # self.seed = seed

        self.rf = None
        self.svm = None
        self.lg = None
        self.knn = None
        self.lda = None

        self.voting = None
        self.pipe = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # np.random.seed(self.seed)

    def update_classifiers_parameters(self, solution: S):
        self.rf = RandomForestClassifier(n_estimators=abs(int(solution.variables[0])),
                                         ccp_alpha=abs(solution.variables[1]),
                                         bootstrap=True, max_features='log2')
        self.svm = SVC(
            C=10 ** int(solution.variables[2]),
            probability=True)

        self.lg = LogisticRegression(solver='saga',
                                     max_iter=abs(int(solution.variables[3])),
                                     C=10 ** int(solution.variables[4]),
                                     tol=10 ** solution.variables[5])

        self.knn = KNeighborsClassifier(n_neighbors=abs(int(solution.variables[6])),
                                        metric='chebyshev', weights='distance')

        self.lda = LinearDiscriminantAnalysis(solver='svd',
                                              tol=10 ** solution.variables[7])

        self.voting = VotingClassifier(
            estimators=[('SVM', self.svm),
                        ('Random Forests', self.rf),
                        ('LogReg', self.lg),
                        ('KNN', self.knn),
                        ('LDA', self.lda)],
            voting='soft')

        # self.relief = ReliefF(n_neighbors=int(solution.variables[8]), n_features_to_keep=int(solution.variables[9]))
        self.kbest = SelectKBest(k=int(solution.variables[8]))

        self.pipe = Pipeline([
            ('feature_selection', self.kbest),
            ('classification', self.voting)

        ])

    def evaluate(self, solution: S) -> S:
        self.update_classifiers_parameters(solution)
        pipeFit = self.pipe.fit(self.X_train, self.y_train)
        solution.attributes[0] = pipeFit.named_steps['feature_selection'].get_support(indices=True)
        predictions = pipeFit.predict_proba(self.X_test)
        score = log_loss(self.y_test, predictions)

        solution.objectives[0] = score

        return solution

    def get_name(self) -> str:
        return 'FS-OptEnsemble'
