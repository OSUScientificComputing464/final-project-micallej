#lighting the hearth
import numpy
import matplotlib.pyplot
import pandas
import csv
import seaborn
import warnings
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
		
def visualizeClassifier(model, X, y, ax=None, cmap='rainbow'):
	
    ax = ax or matplotlib.pyplot.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = numpy.meshgrid(numpy.linspace(*xlim, num=200),numpy.linspace(*ylim, num=200))
    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(numpy.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=numpy.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
		
	
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

ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("all")

matplotlib.pyplot.show()
print("press <enter> to continue")
input()


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

ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("low-temperature only")

matplotlib.pyplot.show()
print("press <enter> to continue")
input()


#Only superconductors with Iron
#cross reference Unique array to find compounds with iron >0
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

ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("Iron only")

matplotlib.pyplot.show()
print("press <enter> to continue")
input()


#Only superconductors with an Oxygen:Copper ratio of 2
#cross reference Unique array to find compounds with iron >0
SuperConductorsDataFrame.loc[(SuperConductorsUnique['Fe'] > 0) & (SuperConductorsUnique['Cu'] > 0) & (round(SuperConductorsUnique['O']/(2*SuperConductorsUnique['Cu']),0)==1)]

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


ColourPlot(AtomicMassData,CriticalTemperatureData)
ColourPlot(FIEData,CriticalTemperatureData)
ColourPlot(AtomicDensityData,CriticalTemperatureData)
ColourPlot(AtomicRadiusData,CriticalTemperatureData)
ColourPlot(ElectronAffinityData,CriticalTemperatureData)
ColourPlot(FusionHeatData,CriticalTemperatureData)
ColourPlot(ThermalConductivityData,CriticalTemperatureData)
ColourPlot(ValenceData,CriticalTemperatureData)

print("HTC only")

matplotlib.pyplot.show()
print("press <enter> to continue")
input()

'''
######   	Random Forrests				#####
XTrain, XTest, YTrain, YTest = train_test_split(SuperConductorsData, SuperConductorsTarget,random_state=0)

model = RandomForestClassifier(n_estimators=1000)
model.fit(XTrain, YTrain)
YPredict = model.predict(XTest)

print(metrics.classification_report(YPredict, YTest))

mat = confusion_matrix(YTest, YPredict)
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
matplotlib.pyplot.xlabel('true label')
matplotlib.pyplot.ylabel('predicted label');
'''

#show plots
matplotlib.pyplot.show()

