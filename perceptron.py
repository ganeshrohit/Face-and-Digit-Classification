import util
PRINT = True

class Perceptron:
    def __init__( self, labels, max_iterations):
        self.labels = labels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in labels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.labels)
        self.weights = weights

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """



        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                real, pred = trainingLabels[i], self.classify([trainingData[i]])[0]
                if real != pred:
                    self.weights[real] += trainingData[i]
                    self.weights[pred] -= trainingData[i]

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.labels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        return self.weights[label].sortedKeys()[:100]