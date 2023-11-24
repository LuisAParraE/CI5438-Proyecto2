import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy

#Definicion de una neurona
class Neuron:

    #Inicializamos los pesos de a neurona con numpy para utilizar la multiplicacion de matrices
    def __init__(self,numberWeights) -> None:
        self.weights = numpy.zeros(numberWeights)
        self.localGradient = None
        self.weightGradient = None
        self.rawValue = None
        self.activatedValue = None
        self.data = None

    #Metodo para calcular el valor de la neurona 
    def calculateValue(self,data):

        #Almacenamos la entrada recibida para poder hacer Backward propagation
        self.data = data

        #calculamos el valor de la neurona en base a los pesos
        self.rawValue = numpy.dot(self.weights.transpose(),data)

        ##Esta es la funcion de activación logistica
        self.activatedValue = 1 /(1 + numpy.exp(-self.rawValue))

        return self.activatedValue
    
    #Calcula los gradientes para utilizarlos para las modificaciones de los pesos de la red
    def calculateGradients(self, upwardGradient,alpha):

        #Multiplicamos la suma de los  upwardGradients por la derivada de la funcion logistica
        self.localGradient = sum(upwardGradient) * self.activatedValue * (1-self.activatedValue)

        #Calcula los gradientes para los pesos
        self.weightGradient = self.data * self.localGradient
        
        self.updateWeights(alpha)
    
    #Funcion para actualizar los pesos
    def updateWeights(self,alpha):
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + alpha * self.weightGradient[i]

#Definicion de una capa
class Layer:

    def __init__(self,inNumber, neuronNumber) -> None:
        self.neuron = []
        
        for i in range(0, neuronNumber):
            self.neuron.append(Neuron(inNumber))
        
        self.preLayer = None
        self.nextLayer = None

    #Funcion que se encarga de modificar los pesos, en base a la perdida.
    def train(self, trainData, trainAnswer,alpha):
        loss = 0

        #Creamos un arreglo con los valores de entrada para la proxima capa
        nextLayerInput = numpy.zeros(len(self.neuron),dtype=float)

        #Obetenemos los valores de entrada de cada neurona
        for i in range(0,len(self.neuron)):
            nextLayerInput[i] = self.neuron[i].calculateValue(trainData)

        #Verificamos si somos la ultima capa, si no lo somos, llamamos al entrenamiento de la proxima capa
        if self.nextLayer is None:

            for j in range(0, len(self.neuron)):
                loss = loss + (trainAnswer[i] - self.neuron[j].activatedValue)**2

            self.backPropagation(trainAnswer,alpha)

            return loss, nextLayerInput
        else:
            return self.nextLayer.train(nextLayerInput,trainAnswer,alpha)

    #Funcion que solo retorna la hipotesis de una entrada
    def test(self, testData, testAnswer):
        loss =0
        nextLayerInput = numpy.zeros(len(self.neuron),dtype=float)
        for i in range(0,len(self.neuron)):
            nextLayerInput[i] = self.neuron[i].calculateValue(testData)

        #Verificamos si somos la ultima capa, si no lo somos, llamamos a la prueba de la proxima capa
        if self.nextLayer is None:
            for j in range(0, len(self.neuron)):
                loss = loss + (testAnswer[i] - self.neuron[j].activatedValue)**2
            return loss,nextLayerInput
        else:
            return self.nextLayer.test(nextLayerInput,testAnswer)

    #Funcion para hacer BackPropagation sobre la red
    def backPropagation(self, trainAnswer,alpha):
        if self.nextLayer is None:
            for i in range(0,len(self.neuron)):
                aux = []
                aux.append((2/len(self.neuron)) * (trainAnswer[i]-self.neuron[i].activatedValue))
                self.neuron[i].calculateGradients(aux,alpha)

            if self.preLayer is not None:
                self.preLayer.backPropagation(trainAnswer,alpha)

        else:
            #Obtenemos todos los gradientes anteriores
            upwardGradient = []
            for i in range(0, len(self.nextLayer.neuron)):
                upwardGradient.append(self.nextLayer.neuron[i].localGradient)

            #Pasamos 
            for i in range(0, len(self.neuron)):
                self.neuron[i].calculateGradients(upwardGradient,alpha)

            #Si no hemos llegado a la raiz, seguimos haciendo la propagacion
            if self.preLayer is not None:
                self.preLayer.backPropagation(trainAnswer,alpha)

#Funcion para cambiar la categoria species a 1 o 0 para cada caso binario
def dataSegmentation(df):

    solSetosa = numpy.zeros(len(df))
    solVersicolor = numpy.zeros(len(df))
    solVirginica = numpy.zeros(len(df))
    solMulticlass = numpy.zeros((len(df),3))
    
    for i in range(0,len(df)):
        if df["species"].iloc[i] == "Iris-setosa":
            solSetosa[i] = 1
            solMulticlass[i][0] = 1
        elif df["species"].iloc[i] == "Iris-versicolor":
            solVersicolor[i] = 1
            solMulticlass[i][1] = 1
        elif df["species"].iloc[i] == "Iris-virginica":
            solVirginica[i] = 1
            solMulticlass[i][2] = 1

    return solSetosa,solVersicolor,solVirginica,solMulticlass

def NNCreation():
    print()

def fastNNCreation(inValueQuant: int, numberLayers: int, neuronForLayer: list[int]):

    allLayers = []

    if inValueQuant < 0:
        print("Error Invalid In Quantity")
        return -1
    elif numberLayers < 0:
        print ("Error Invalid Number of Layers")
        return -2

    #Inicializa las neuronas en cada capa
    for i in range(0,numberLayers):
        if neuronForLayer[i] < 0:
            print ("Error Invalid Number of Neurons")
            return -2
        
        #Si la capa es la primera en crearse, usa la cantidad de parametros de entrada, sino, la cantidad de neuronas de 
        # la capa anterior
        if i == 0:
            allLayers.append(Layer(inValueQuant,neuronForLayer[i]))
        else:
            allLayers.append(Layer(neuronForLayer[i-1],neuronForLayer[i]))
        
    #Asigna la referencias a las capas entre ellas
    for i in range(1,len(allLayers)):
        allLayers[i].preLayer = allLayers[i-1]
        allLayers[i-1].nextLayer = allLayers[i]
    
    neuralNet = allLayers[0]

    return neuralNet

def beginTraining(NN:Layer, epoch:int, learningRate:float ,dataTrain, dataTrainAnswer, dataTest, dataTestAnswer):

    testLoss = []
    trainLoss= []
    bestModel = None
    bestModelLayer = None
    #Hacemos un entrenamiento por epocas
    for j in range(0,epoch):
        auxLoss = 0
        #Le vamos pasando cada fila de datos al modelo
        for i in range(0,len(dataTrain)):
            if dataTrainAnswer.ndim ==1:
                auxLoss1,auxValues = NN.train(trainData=dataTrain[i],trainAnswer=dataTrainAnswer[i:],alpha= learningRate)
            else:
                auxLoss1,auxValues = NN.train(trainData=dataTrain[i],trainAnswer=dataTrainAnswer[i], alpha= learningRate)

            auxLoss = auxLoss + auxLoss1

        #Guardamos la Mejor Red Neuronal hasta el momento
        if bestModel is None or auxLoss < bestModel:
            bestModel = auxLoss
            bestModelLayer = copy.deepcopy(NN)
        
        #Guardamos el arreglo de perdidas por epoca
        trainLoss.append(auxLoss)
        auxLoss = 0

        #Aca verificamos los datos del conjunto de pruebas 
        for i in range(0,len(dataTest)):
            if dataTestAnswer.ndim ==1:
                auxLoss1,auxValues = NN.test(testData=dataTest[i],testAnswer=dataTestAnswer[i:])
            else:
                auxLoss1,auxValues = NN.test(testData=dataTest[i],testAnswer=dataTestAnswer[i])

            auxLoss = auxLoss + auxLoss1
        testLoss.append(auxLoss)

    return trainLoss, testLoss, bestModelLayer

def hipAproximation(trustGrade:float, NN:Layer,data, dataAnswer):

    falsePos = 0
    truePos = 0
    falseNeg = 0
    trueNeg = 0

    falsePosRate = 0
    falseNegRate = 0

    hipotesis = []
    simpleAnswer = None

    for i in range(0,len(dataAnswer)):
        loss , simpleAnswer=NN.test(data[i],dataAnswer[i:])
        hipotesis.append(simpleAnswer)

    for i in range(0,len(dataAnswer)):
        if hipotesis[i] >= trustGrade:
            aux = 1
        else:
            aux = 0
        #print(f"hip:{aux}  realVal:{dataAnswer[i]}")
        if aux == dataAnswer[i] and aux == 1:
            #print("True Pos")
            truePos = truePos + 1
        elif aux == dataAnswer[i] and aux == 0:
            #print("True Neg")
            trueNeg = trueNeg +1
        elif aux != dataAnswer[i] and aux == 1:
            #print("False Pos")
            falsePos = falsePos +1
        elif aux != dataAnswer[i] and aux == 0:
            #print("False Neg")
            falseNeg = falseNeg +1

    #falsePosRate = falsePos/(falsePos+truePos)
    #falseNegRate = falseNeg/(falseNeg+trueNeg)

    return falsePos,falseNeg

def main():
    case = 2
    if case ==1:
        #Se lee el CSV con pandas
        dfPlantas = pandas.read_csv('iris.csv')

        #Parametrizamos los valores numericos
        dfPlantas["sepal_length"] = (dfPlantas["sepal_length"] - dfPlantas["sepal_length"].min()) / (dfPlantas["sepal_length"].max() - dfPlantas["sepal_length"].min())
        dfPlantas["sepal_width"] = (dfPlantas["sepal_width"] - dfPlantas["sepal_width"].min()) / (dfPlantas["sepal_width"].max() - dfPlantas["sepal_width"].min())
        dfPlantas["petal_length"] = (dfPlantas["petal_length"] - dfPlantas["petal_length"].min()) / (dfPlantas["petal_length"].max() - dfPlantas["petal_length"].min())
        dfPlantas["petal_width"] = (dfPlantas["petal_width"] - dfPlantas["petal_width"].min()) / (dfPlantas["petal_width"].max() - dfPlantas["petal_width"].min())

        dfDatosBin = dfPlantas.drop(columns=["species"])

        #--------------Segmentación de datos Parte 2---------------------#

        #Obtenemos un arreglo de 0 y 1 para cada tipo de flor
        binSolSetosa,binSolVersicolor,binSolVirginica, multiSolPlants = dataSegmentation(dfPlantas)

        #Hacemos el Cross Data Validation para las setosas
        
        auxDataTraining,auxDataTest,auxAnswerTraining,auxAnswerTest = train_test_split(
            dfDatosBin, binSolSetosa,test_size= 0.2,shuffle=True
        )

        #Lo pasamos todo a numpy para aprovechar la multiplicacion de matrices
        dataTrainingSetosa = numpy.array(auxDataTraining)
        dataTestSetosa = numpy.array(auxDataTest)
        answerTrainingSetosa = numpy.array(auxAnswerTraining)
        answerTestSetosa = numpy.array(auxAnswerTest)

        #Hacemos el Cross Data Validation para las Versicolor
        auxDataTraining,auxDataTest,auxAnswerTraining,auxAnswerTest = train_test_split(
            dfDatosBin, binSolVersicolor,test_size= 0.2,shuffle=True
        )

        #Lo pasamos todo a numpy para aprovechar la multiplicacion de matrices
        dataTrainingVersicolor = numpy.array(auxDataTraining)
        dataTestVersicolor = numpy.array(auxDataTest)
        answerTrainingVersicolor = numpy.array(auxAnswerTraining)
        answerTestVersicolor = numpy.array(auxAnswerTest)

        #Hacemos el Cross Data Validation para las Virginicas
        auxDataTraining,auxDataTest,auxAnswerTraining,auxAnswerTest = train_test_split(
            dfDatosBin, binSolVirginica,test_size= 0.2,shuffle=True
        )

        #Lo pasamos todo a numpy para aprovechar la multiplicacion de matrices
        dataTrainingVirginica = numpy.array(auxDataTraining)
        dataTestVirginica = numpy.array(auxDataTest)
        answerTrainingVirginica = numpy.array(auxAnswerTraining)
        answerTestVirginica = numpy.array(auxAnswerTest)
        
        #Hacemos el Cross Data Validation para el multiclass
        auxDataTraining,auxDataTest,auxAnswerTraining,auxAnswerTest = train_test_split(
            dfDatosBin, multiSolPlants,test_size= 0.2,shuffle=True
        )

        #Lo pasamos todo a numpy para aprovechar la multiplicacion de matrices
        dataTrainingMulticlass = numpy.array(auxDataTraining)
        dataTestMulticlass = numpy.array(auxDataTest)
        answerTrainingMulticlass = numpy.array(auxAnswerTraining)
        answerTestMulticlass = numpy.array(auxAnswerTest)

        #Creamos la red neuronal
        nnTest = fastNNCreation(4,3,[1,2,1])
        alpha = 0.01
        epoch = 5000

        lossTrain,lossTest, bestNN = beginTraining(nnTest,epoch,alpha,dataTrainingSetosa,answerTrainingSetosa,dataTestSetosa,answerTestSetosa)
        falsePosTrain,falseNegTrain = hipAproximation(0.90,bestNN,dataTrainingSetosa,answerTrainingSetosa)
        falsePosTest,falseNegTest= hipAproximation(0.90,bestNN,dataTestSetosa,answerTestSetosa)
        
        meanLossTrain = [x/120 for x in lossTrain]
        meanLossTest = [x/30 for x in lossTest]

        print(f"Max Mean Train Loss:{max(lossTrain)/120}")
        print(f"Max Mean Test Loss:{max(lossTest)/30}")
        print(f"Min Mean Train Loss:{min(lossTrain)/120}")
        print(f"Min Mean Test Loss:{min(lossTest)/30}")
        print(f"False Positive Train Count:{falsePosTrain}")
        print(f"False Negative Train Count:{falseNegTrain}")
        print(f"False Positive Test Count:{falsePosTest}")
        print(f"False Negative Test Count:{falseNegTest}")

        #plt.plot(lossTrain)
        #plt.title(f"Train loss with alpha:{alpha}")
        #plt.ylabel("Loss")
        #plt.xlabel("Epoch")
        #plt.show()

        #plt.plot(lossTest)
        #plt.title(f"Test loss with alpha:{alpha}")
        #plt.ylabel("Loss")
        #plt.xlabel("Epoch")
        #plt.show()

        plt.plot(meanLossTrain)
        plt.title(f"Mean Train loss with alpha:{alpha}")
        plt.ylabel("Mean Loss")
        plt.xlabel("Epoch")
        plt.show()

        plt.plot(meanLossTest)
        plt.title(f"Mean Test loss with alpha:{alpha}")
        plt.ylabel("Mean Loss")
        plt.xlabel("Epoch")
        plt.show()

    else:
        dfSpam = pandas.read_csv('spambase/spambase.csv',decimal=".",header=0)
        dfSpam.astype(float)
        dfSpamBin = dfSpam.drop(columns=["is_spam"])
        isSpamCol = dfSpam["is_spam"]

        """ spam = dfSpam[dfSpam['spam_or_not'] == 1]
        ham = dfSpam[dfSpam['spam_or_not'] == 0]

        spam_train, spam_test = train_test_split(spam, train_size=0.7)
        ham_train, ham_test = train_test_split(ham,train_size=0.7)

        X_train = ham_train._append(spam_train)
        y_train = X_train.pop('spam_or_not')

        X_test = ham_test._append(spam_test)
        y_test = X_test.pop('spam_or_not')

        print(X_train)

        print(y_train)"""
        # Normalizamos todas las columnas
        for elem in dfSpamBin:
            dfSpamBin[elem] = pandas.to_numeric(dfSpamBin[elem])
        for elem in dfSpamBin:
            dfSpamBin[elem] = (dfSpamBin[elem] - dfSpamBin[elem].min())/(dfSpamBin[elem].max() - dfSpamBin[elem].min())
        #dfSpamNormalized = (dfSpamBin - dfSpamBin.min())/(dfSpamBin.max()-dfSpamBin.min())
        auxDataTraining, auxDataTest, auxAnswerTraining, auxAnswerTest = train_test_split(
            dfSpamBin, isSpamCol,test_size= 0.3,shuffle=True
        )
        
        dataTraining = numpy.array(auxDataTraining)
        dataTest = numpy.array(auxDataTest)
        answerTraining = numpy.array(auxAnswerTraining)
        answerTest = numpy.array(auxAnswerTest)
        capas = 2
        capasNeurona = [4,1]
        alpha = 0.01
        learningRate = 0.70
        for i in [500,1000,2500,5000]:
            epoch = i
            nnTest = fastNNCreation(57,capas,capasNeurona)
            lossTrain,lossTest, bestNN = beginTraining(nnTest,epoch,alpha,dataTraining,answerTraining,dataTest,answerTest)
            falsePosTrain,falseNegTrain = hipAproximation(learningRate,bestNN,dataTraining,answerTraining)
            falsePosTest,falseNegTest= hipAproximation(learningRate,bestNN,dataTest,answerTest)
            meanLossTrain = [x/len(dataTraining) for x in lossTrain]
            meanLossTest = [x/len(dataTest) for x in lossTest]
            """ print(f"Max Mean Train Loss:{max(lossTrain)/len(dataTraining)}")
            print(f"Max Mean Test Loss:{max(lossTest)/len(dataTest)}")
            print(f"Min Mean Train Loss:{min(lossTrain)/len(dataTraining)}")
            print(f"Min Mean Test Loss:{min(lossTest)/len(dataTest)}")
            print(f"False Positive Train Count:{falsePosTrain}")
            print(f"False Negative Train Count:{falseNegTrain}")
            print(f"False Positive Test Count:{falsePosTest}")
            print(f"False Negative Test Count:{falseNegTest}") """
            print(f"{epoch},{capas},{capasNeurona},{learningRate},{min(lossTrain)/len(dataTraining)},{max(lossTrain)/len(dataTraining)},{falsePosTrain},{falseNegTrain},{min(lossTest)/len(dataTest)},{max(lossTest)/len(dataTest)},{falseNegTest},{falsePosTest}",file=open('outputs.csv', 'a'))
            plt.plot(meanLossTrain)
            plt.title(f"Mean Train loss with alpha:{alpha} and {capasNeurona} in {capas} capes - LR: {learningRate}")
            plt.ylabel("Mean Loss")
            plt.xlabel("Epoch")
            plt.savefig(f'capas_{capas}_capasN_{capasNeurona}_mean_train_alpha_{alpha}_epoch_{epoch}_LR_{learningRate}.png')
            plt.close()
            plt.plot(meanLossTest)
            plt.title(f"Mean Test loss with alpha:{alpha} and {capasNeurona} in {capas} capes - LR: {learningRate}")
            plt.ylabel("Mean Loss")
            plt.xlabel("Epoch")
            plt.savefig(f'capas_{capas}_capasN_{capasNeurona}_mean_test_alpha_{alpha}_epoch_{epoch}_LR_{learningRate}.png')

main()