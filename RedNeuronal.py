import pandas
import numpy

#Learnig rate
alpha = 0.001

#Definicion de una neurona
class neuron:
    weights = []
    localHip = None
    value = None

    #Inicializamos los pesos de a neuroana con numpy para utilizar la multipliacion de matrices
    def weightInit(self,newWeights):
        self.weights = numpy.array(newWeights)

    def localCalculation():
        print()

    #Metodo para calcular el valor de la neurona 
    def calculateValue(self,data):
        #calculamos el valor de la neurona en base a los pesos
        tempValue = numpy.dot(self.weights.transpose(),data)

        ##Esta es la funcion de activación logistica
        self.value = 1 /(1 + numpy.e ** (-tempValue))

        return self.value

#Definicion de una capa
class layer:
    neuron = []
    preLayer = None
    nextLayer = None

    def train(self, data, answer):
        print("UwU")
    
    def test():
        print("UwU")

    def backPropagation():
        print("UwU")


def main():
    print("UwU")