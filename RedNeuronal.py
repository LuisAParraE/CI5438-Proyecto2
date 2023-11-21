import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Learnig rate

#Definicion de una neurona
class neuron:

    #Inicializamos los pesos de a neurona con numpy para utilizar la multiplicacion de matrices
    def __init__(self,numberWeights) -> None:
        self.weights = numpy.zeros(numberWeights)
        self.localGradient = None
        self.weightGradient = None
        self.rawValue = None
        self.activatedValue = None
        self.data = None

    def localCalculation():
        print()

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
class layer:

    def __init__(self,inNumber, neuronNumber) -> None:
        self.neuron = []
        
        for i in range(0, neuronNumber):
            self.neuron.append(neuron(inNumber))
        
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
            allLayers.append(layer(inValueQuant,neuronForLayer[i]))
        else:
            allLayers.append(layer(neuronForLayer[i-1],neuronForLayer[i]))
        
    #Asigna la referencias a las capas entre ellas
    for i in range(1,len(allLayers)):
        allLayers[i].preLayer = allLayers[i-1]
        allLayers[i-1].nextLayer = allLayers[i]
    
    neuralNet = allLayers[0]

    return neuralNet

def beginTraining(NN:layer, epoch:int, learningRate:float ,dataTrain, dataTrainAnswer, dataTest, dataTestAnswer):

    testLoss = []
    trainLoss= []
    for j in range(0,epoch):
        auxLoss = 0
        for i in range(0,len(dataTrain)):
            if dataTrainAnswer.ndim ==1:
                auxLoss1,auxValues = NN.train(trainData=dataTrain[i],trainAnswer=dataTrainAnswer[i:],alpha= learningRate)
            else:
                auxLoss1,auxValues = NN.train(trainData=dataTrain[i],trainAnswer=dataTrainAnswer[i], alpha= learningRate)
            auxLoss = auxLoss + auxLoss1
        trainLoss.append(auxLoss)
        auxLoss = 0
        for i in range(0,len(dataTest)):
            if dataTestAnswer.ndim ==1:
                auxLoss1,auxValues = NN.test(testData=dataTest[i],testAnswer=dataTestAnswer[i:])
            else:
                auxLoss1,auxValues = NN.test(testData=dataTest[i],testAnswer=dataTestAnswer[i])

            auxLoss = auxLoss + auxLoss1
        testLoss.append(auxLoss)

    return trainLoss, testLoss

def main():
    case = 1
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
        nn = fastNNCreation(5,3,[3,5,2])
        print(len(nn.neuron))
        print(len(nn.neuron[0].weights))
        print(len(nn.neuron[1].weights))
        print(len(nn.nextLayer.neuron))
        print(len(nn.nextLayer.neuron[0].weights))
        print(len(nn.nextLayer.neuron[1].weights))

        nnTest = fastNNCreation(4,2,[2,1])

        lossTrain,lossTest = beginTraining(nnTest,5000,0.1,dataTrainingSetosa,answerTrainingSetosa,dataTestSetosa,answerTestSetosa)

        print(lossTrain[len(lossTest)-1])
        print(lossTest[len(lossTest)-1])
        print(len(lossTrain))

    else:
        print()
    #---------------- Aca se desarrolla la parte 3---------------------#


main()