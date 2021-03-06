#lighting the hearth
import numpy
import matplotlib.pyplot
import pandas
import csv
import seaborn
import warnings
import os
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#functions
def GridPlot(X,Y):
	#plot each column of X on a subplot
	i=1
	for Name in X.columns:
		#matplotlib.pyplot.subplot(numpy.ceil(numpy.sqrt(X.shape[1])),numpy.ceil(numpy.sqrt(X.shape[1])),i)
		matplotlib.pyplot.subplot(X.shape[1],1,i)
		matplotlib.pyplot.scatter(Y[:],X.loc[:,Name])
		matplotlib.pyplot.ylabel(Name,rotation=0)
		matplotlib.pyplot.xlabel("critical temperature",rotation=0)
		i+=1
		
def ColourPlot(X,Y):
	#plot each column of X on the same figure with a different colour
	matplotlib.pyplot.figure()
	colour = numpy.array(['red','magenta','green','gold','blue','cyan','orange','violet','black','grey'])
	
	matplotlib.pyplot.ylabel("data",rotation=0)
	matplotlib.pyplot.xlabel("critical temperature",rotation=0)
	i=0
	for Name in X.columns:
		matplotlib.pyplot.scatter(Y[:],X.loc[:,Name],c=colour[i],label=Name)
		i+=1
	matplotlib.pyplot.legend(loc='upper right')
	
def ISOPlot(X, Y, n):
    iso = Isomap(n_components=2)
    projected = iso.fit_transform(X)
    matplotlib.pyplot.figure(figsize = (12,8))
    matplotlib.pyplot.scatter(projected[:, 0], projected[:, 1],c=Y, edgecolor='none', alpha=0.5,cmap=matplotlib.pyplot.cm.get_cmap('nipy_spectral', 10))
    matplotlib.pyplot.title('Isomap Reduction n='+str(n), fontsize=16)
    matplotlib.pyplot.xlabel('component 1', fontsize=16)
    matplotlib.pyplot.ylabel('component 2', fontsize=16)
    matplotlib.pyplot.colorbar();

def PCAPlot(X, Y, n):
    pca = PCA(n_components = n)
    projected = pca.fit_transform(X)

    matplotlib.pyplot.figure(figsize = (12,8))
    matplotlib.pyplot.scatter(projected[:, 0], projected[:, 1],c=Y, edgecolor='none', alpha=0.5,cmap=matplotlib.pyplot.cm.get_cmap('nipy_spectral', 10))
    matplotlib.pyplot.title('PCA Reduction n='+str(n), fontsize=16)
    matplotlib.pyplot.xlabel('component 1', fontsize=16)
    matplotlib.pyplot.ylabel('component 2', fontsize=16)
    matplotlib.pyplot.colorbar();
    
    print('The  values represent the importance of each of the component axes in classifying the dataset:')
    print('explained variance: ', pca.explained_variance_)	
	
def RFScore(XTrain, XTest, YTrain, YTest, n):
	#perform random forrst learning and score result
    scores = []
    for i in range(n):
        RFC = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        RFC.fit(XTrain,YTrain)
        scores.append(RFC.score(XTest,YTest))
    return numpy.mean(scores)
    
    
#preperations

warnings.filterwarnings("ignore", category=DeprecationWarning) 
seaborn.set()

SuperConductorsCSV = numpy.loadtxt('predict_tc-master/train.csv',delimiter=',',skiprows=1)
SuperConductorsFeatures = numpy.array(['number_of_elements',
										'mean_atomic_mass','weighted_mean_atomic_mass','gmean_atomic_mass','weighted_gmean_atomic_mass','entropy_atomic_mass','weighted_entropy_atomic_mass','range_atomic_mass','weighted_range_atomic_mass','standard_atomic_mass','weighted_standard_atomic_mass',
										'mean_fie','weighted_mean_fie','gmean_fie','weighted_gmean_fie','entropy_fie','weighted_entropy_fie','range_fie','weighted_range_fie','standard_fie','weighted_standard_fie',
										'mean_atomic_radius','weighted_mean_atomic_radius','gmean_atomic_radius','weighted_gmean_atomic_radius','entropy_atomic_radius','weighted_entropy_atomic_radius','range_atomic_radius','weighted_range_atomic_radius','standard_atomic_radius','weighted_standard_atomic_radius',
										'mean_density','weighted_mean_density','gmean_density','weighted_gmean_density','entropy_density','weighted_entropy_density','range_density','weighted_range_density','standard_density','weighted_standard_density',
										'mean_electron_affinity','weighted_mean_electron_affinity','gmean_electron_affinity','weighted_gmean_electron_affinity','entropy_electron_affinity','weighted_entropy_electron_affinity','range_electron_affinity','weighted_range_electron_affinity','standard_electron_affinity','weighted_standard_electron_affinity',
										'mean_fusion_heat','weighted_mean_fusion_heat','gmean_fusion_heat','weighted_gmean_fusion_heat','entropy_fusion_heat','weighted_entropy_fusion_heat','range_fusion_heat','weighted_range_fusion_heat','standard_fusion_heat','weighted_standard_fusion_heat',
										'mean_thermal_conductivity','weighted_mean_thermal_conductivity','gmean_thermal_conductivity','weighted_gmean_thermal_conductivity','entropy_thermal_conductivity','weighted_entropy_thermal_conductivity','range_thermal_conductivity','weighted_range_thermal_conductivity','standard_thermal_conductivity','weighted_standard_thermal_conductivity',
										'mean_valence','weighted_mean_valence','gmean_valence','weighted_gmean_valence','entropy_valence','weighted_entropy_valence','range_valence','weighted_range_valence','standard_valence','weighted_standard_valence',
										'critical_temperature'])
										
SuperConductorsFormulaCSV = numpy.loadtxt('predict_tc-master/unique.csv',delimiter=',',skiprows=1,usecols=range(0,87))
SuperConductorsChemicals = numpy.array(['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','critical_temp'])

SuperConductorsDataFrame = pandas.DataFrame(data=SuperConductorsCSV,columns=SuperConductorsFeatures)
SuperConductorsUnique = pandas.DataFrame(data=SuperConductorsFormulaCSV,columns=SuperConductorsChemicals)

SuperConductorsData = SuperConductorsDataFrame.loc[:,SuperConductorsDataFrame.columns !='critical_temperature']
SuperConductorsTarget = SuperConductorsDataFrame.loc[:,'critical_temperature']

scores =[]


######   	Visualize Data				#####
#full set
NumberOfElementsData = SuperConductorsDataFrame.loc[:,('number_of_elements')]
AtomicMassData = SuperConductorsDataFrame.loc[:,('mean_atomic_mass','weighted_mean_atomic_mass','gmean_atomic_mass','weighted_gmean_atomic_mass','entropy_atomic_mass','weighted_entropy_atomic_mass','range_atomic_mass','weighted_range_atomic_mass','standard_atomic_mass','weighted_standard_atomic_mass')]
FIEData = SuperConductorsDataFrame.loc[:,('mean_fie','weighted_mean_fie','gmean_fie','weighted_gmean_fie','entropy_fie','weighted_entropy_fie','range_fie','weighted_range_fie','standard_fie','weighted_standard_fie',)]
AtomicRadiusData = SuperConductorsDataFrame.loc[:,('mean_atomic_radius','weighted_mean_atomic_radius','gmean_atomic_radius','weighted_gmean_atomic_radius','entropy_atomic_radius','weighted_entropy_atomic_radius','range_atomic_radius','weighted_range_atomic_radius','standard_atomic_radius','weighted_standard_atomic_radius')]
AtomicDensityData = SuperConductorsDataFrame.loc[:,('mean_density','weighted_mean_density','gmean_density','weighted_gmean_density','entropy_density','weighted_entropy_density','range_density','weighted_range_density','standard_density','weighted_standard_density')]
ElectronAffinityData = SuperConductorsDataFrame.loc[:,('mean_electron_affinity','weighted_mean_electron_affinity','gmean_electron_affinity','weighted_gmean_electron_affinity','entropy_electron_affinity','weighted_entropy_electron_affinity','range_electron_affinity','weighted_range_electron_affinity','standard_electron_affinity','weighted_standard_electron_affinity',)]
FusionHeatData = SuperConductorsDataFrame.loc[:,('mean_fusion_heat','weighted_mean_fusion_heat','gmean_fusion_heat','weighted_gmean_fusion_heat','entropy_fusion_heat','weighted_entropy_fusion_heat','range_fusion_heat','weighted_range_fusion_heat','standard_fusion_heat','weighted_standard_fusion_heat')]
ThermalConductivityData = SuperConductorsDataFrame.loc[:,('mean_thermal_conductivity','weighted_mean_thermal_conductivity','gmean_thermal_conductivity','weighted_gmean_thermal_conductivity','entropy_thermal_conductivity','weighted_entropy_thermal_conductivity','range_thermal_conductivity','weighted_range_thermal_conductivity','standard_thermal_conductivity','weighted_standard_thermal_conductivity')]
ValenceData = SuperConductorsDataFrame.loc[:,('mean_valence','weighted_mean_valence','gmean_valence','weighted_gmean_valence','entropy_valence','weighted_entropy_valence','range_valence','weighted_range_valence','standard_valence','weighted_standard_valence')]
CriticalTemperatureData = SuperConductorsDataFrame.loc[:,'critical_temperature']

print("showing all")
ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)


print("iso all")
ISOPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)
print("pca all")
PCAPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)

matplotlib.pyplot.show()
print("press <enter> to continue")
input()
os.system('cls' if os.name == 'nt' else 'clear')


#Only values with Tc below  10K 
#keep all values are below 10K
SuperConductorsDataFrame.loc[SuperConductorsDataFrame['critical_temperature'] < 10]

NumberOfElementsData = SuperConductorsDataFrame.loc[:,('number_of_elements')]
AtomicMassData = SuperConductorsDataFrame.loc[:,('mean_atomic_mass','weighted_mean_atomic_mass','gmean_atomic_mass','weighted_gmean_atomic_mass','entropy_atomic_mass','weighted_entropy_atomic_mass','range_atomic_mass','weighted_range_atomic_mass','standard_atomic_mass','weighted_standard_atomic_mass')]
FIEData = SuperConductorsDataFrame.loc[:,('mean_fie','weighted_mean_fie','gmean_fie','weighted_gmean_fie','entropy_fie','weighted_entropy_fie','range_fie','weighted_range_fie','standard_fie','weighted_standard_fie',)]
AtomicRadiusData = SuperConductorsDataFrame.loc[:,('mean_atomic_radius','weighted_mean_atomic_radius','gmean_atomic_radius','weighted_gmean_atomic_radius','entropy_atomic_radius','weighted_entropy_atomic_radius','range_atomic_radius','weighted_range_atomic_radius','standard_atomic_radius','weighted_standard_atomic_radius')]
AtomicDensityData = SuperConductorsDataFrame.loc[:,('mean_density','weighted_mean_density','gmean_density','weighted_gmean_density','entropy_density','weighted_entropy_density','range_density','weighted_range_density','standard_density','weighted_standard_density')]
ElectronAffinityData = SuperConductorsDataFrame.loc[:,('mean_electron_affinity','weighted_mean_electron_affinity','gmean_electron_affinity','weighted_gmean_electron_affinity','entropy_electron_affinity','weighted_entropy_electron_affinity','range_electron_affinity','weighted_range_electron_affinity','standard_electron_affinity','weighted_standard_electron_affinity',)]
FusionHeatData = SuperConductorsDataFrame.loc[:,('mean_fusion_heat','weighted_mean_fusion_heat','gmean_fusion_heat','weighted_gmean_fusion_heat','entropy_fusion_heat','weighted_entropy_fusion_heat','range_fusion_heat','weighted_range_fusion_heat','standard_fusion_heat','weighted_standard_fusion_heat')]
ThermalConductivityData = SuperConductorsDataFrame.loc[:,('mean_thermal_conductivity','weighted_mean_thermal_conductivity','gmean_thermal_conductivity','weighted_gmean_thermal_conductivity','entropy_thermal_conductivity','weighted_entropy_thermal_conductivity','range_thermal_conductivity','weighted_range_thermal_conductivity','standard_thermal_conductivity','weighted_standard_thermal_conductivity')]
ValenceData = SuperConductorsDataFrame.loc[:,('mean_valence','weighted_mean_valence','gmean_valence','weighted_gmean_valence','entropy_valence','weighted_entropy_valence','range_valence','weighted_range_valence','standard_valence','weighted_standard_valence')]
CriticalTemperatureData = SuperConductorsDataFrame.loc[:,'critical_temperature']

print("showing low-temperature only")

ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("iso low temperature")
ISOPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)
print("pca low temperature")
PCAPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)


matplotlib.pyplot.show()
print("press <enter> to continue")
input()
os.system('cls' if os.name == 'nt' else 'clear')

#Only superconductors with Iron
#cross reference Unique array to find compounds with iron > 0
SuperConductorsDataFrame.loc[SuperConductorsUnique['Fe'] > 0]

NumberOfElementsData = SuperConductorsDataFrame.loc[:,('number_of_elements')]
AtomicMassData = SuperConductorsDataFrame.loc[:,('mean_atomic_mass','weighted_mean_atomic_mass','gmean_atomic_mass','weighted_gmean_atomic_mass','entropy_atomic_mass','weighted_entropy_atomic_mass','range_atomic_mass','weighted_range_atomic_mass','standard_atomic_mass','weighted_standard_atomic_mass')]
FIEData = SuperConductorsDataFrame.loc[:,('mean_fie','weighted_mean_fie','gmean_fie','weighted_gmean_fie','entropy_fie','weighted_entropy_fie','range_fie','weighted_range_fie','standard_fie','weighted_standard_fie',)]
AtomicRadiusData = SuperConductorsDataFrame.loc[:,('mean_atomic_radius','weighted_mean_atomic_radius','gmean_atomic_radius','weighted_gmean_atomic_radius','entropy_atomic_radius','weighted_entropy_atomic_radius','range_atomic_radius','weighted_range_atomic_radius','standard_atomic_radius','weighted_standard_atomic_radius')]
AtomicDensityData = SuperConductorsDataFrame.loc[:,('mean_density','weighted_mean_density','gmean_density','weighted_gmean_density','entropy_density','weighted_entropy_density','range_density','weighted_range_density','standard_density','weighted_standard_density')]
ElectronAffinityData = SuperConductorsDataFrame.loc[:,('mean_electron_affinity','weighted_mean_electron_affinity','gmean_electron_affinity','weighted_gmean_electron_affinity','entropy_electron_affinity','weighted_entropy_electron_affinity','range_electron_affinity','weighted_range_electron_affinity','standard_electron_affinity','weighted_standard_electron_affinity',)]
FusionHeatData = SuperConductorsDataFrame.loc[:,('mean_fusion_heat','weighted_mean_fusion_heat','gmean_fusion_heat','weighted_gmean_fusion_heat','entropy_fusion_heat','weighted_entropy_fusion_heat','range_fusion_heat','weighted_range_fusion_heat','standard_fusion_heat','weighted_standard_fusion_heat')]
ThermalConductivityData = SuperConductorsDataFrame.loc[:,('mean_thermal_conductivity','weighted_mean_thermal_conductivity','gmean_thermal_conductivity','weighted_gmean_thermal_conductivity','entropy_thermal_conductivity','weighted_entropy_thermal_conductivity','range_thermal_conductivity','weighted_range_thermal_conductivity','standard_thermal_conductivity','weighted_standard_thermal_conductivity')]
ValenceData = SuperConductorsDataFrame.loc[:,('mean_valence','weighted_mean_valence','gmean_valence','weighted_gmean_valence','entropy_valence','weighted_entropy_valence','range_valence','weighted_range_valence','standard_valence','weighted_standard_valence')]
CriticalTemperatureData = SuperConductorsDataFrame.loc[:,'critical_temperature']

print("showing Iron containing only")

ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("iso iron")
ISOPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)
print("pca iron")
PCAPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)


matplotlib.pyplot.show()
print("press <enter> to continue")
input()
os.system('cls' if os.name == 'nt' else 'clear')

#Only superconductors with an Oxygen:Copper ratio of 2
#cross reference Unique array to find compounds with oxygen > 0
#cross reference Unique array to find compounds with copper > 0
#cross reference Unique array to find compounds with oxygen/copper~2
SuperConductorsDataFrame.loc[(SuperConductorsUnique['O'] > 0) & (SuperConductorsUnique['Cu'] > 0) & (round(SuperConductorsUnique['O']/(2*SuperConductorsUnique['Cu']),0)==1)]

NumberOfElementsData = SuperConductorsDataFrame.loc[:,('number_of_elements')]
AtomicMassData = SuperConductorsDataFrame.loc[:,('mean_atomic_mass','weighted_mean_atomic_mass','gmean_atomic_mass','weighted_gmean_atomic_mass','entropy_atomic_mass','weighted_entropy_atomic_mass','range_atomic_mass','weighted_range_atomic_mass','standard_atomic_mass','weighted_standard_atomic_mass')]
FIEData = SuperConductorsDataFrame.loc[:,('mean_fie','weighted_mean_fie','gmean_fie','weighted_gmean_fie','entropy_fie','weighted_entropy_fie','range_fie','weighted_range_fie','standard_fie','weighted_standard_fie',)]
AtomicRadiusData = SuperConductorsDataFrame.loc[:,('mean_atomic_radius','weighted_mean_atomic_radius','gmean_atomic_radius','weighted_gmean_atomic_radius','entropy_atomic_radius','weighted_entropy_atomic_radius','range_atomic_radius','weighted_range_atomic_radius','standard_atomic_radius','weighted_standard_atomic_radius')]
AtomicDensityData = SuperConductorsDataFrame.loc[:,('mean_density','weighted_mean_density','gmean_density','weighted_gmean_density','entropy_density','weighted_entropy_density','range_density','weighted_range_density','standard_density','weighted_standard_density')]
ElectronAffinityData = SuperConductorsDataFrame.loc[:,('mean_electron_affinity','weighted_mean_electron_affinity','gmean_electron_affinity','weighted_gmean_electron_affinity','entropy_electron_affinity','weighted_entropy_electron_affinity','range_electron_affinity','weighted_range_electron_affinity','standard_electron_affinity','weighted_standard_electron_affinity',)]
FusionHeatData = SuperConductorsDataFrame.loc[:,('mean_fusion_heat','weighted_mean_fusion_heat','gmean_fusion_heat','weighted_gmean_fusion_heat','entropy_fusion_heat','weighted_entropy_fusion_heat','range_fusion_heat','weighted_range_fusion_heat','standard_fusion_heat','weighted_standard_fusion_heat')]
ThermalConductivityData = SuperConductorsDataFrame.loc[:,('mean_thermal_conductivity','weighted_mean_thermal_conductivity','gmean_thermal_conductivity','weighted_gmean_thermal_conductivity','entropy_thermal_conductivity','weighted_entropy_thermal_conductivity','range_thermal_conductivity','weighted_range_thermal_conductivity','standard_thermal_conductivity','weighted_standard_thermal_conductivity')]
ValenceData = SuperConductorsDataFrame.loc[:,('mean_valence','weighted_mean_valence','gmean_valence','weighted_gmean_valence','entropy_valence','weighted_entropy_valence','range_valence','weighted_range_valence','standard_valence','weighted_standard_valence')]
CriticalTemperatureData = SuperConductorsDataFrame.loc[:,'critical_temperature']

print("showing HTC only")

ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("iso htc")
ISOPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)
print("pca htc")
PCAPlot(SuperConductorsDataFrame,CriticalTemperatureData,2)


matplotlib.pyplot.show()
print("press <enter> to continue")
input()


######   	Random Forrests				#####
XTrain, XTest, YTrain, YTest = train_test_split(SuperConductorsData, SuperConductorsTarget,random_state=0)

score = RFScore(XTrain,XTest,YTrain,YTest,20)
print("average score: ",score)
