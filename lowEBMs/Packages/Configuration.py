"""Package for importing system configuration.
   The needed data is imported from a specific config.ini and placed in different arrays
   which can be called """
import configparser
import os
from Functions import *
from Variables import *

def importer(path,filename): 
    
    #Importing the configfile.ini
    config=configparser.ConfigParser() 
    os.chdir(path)                     
    config.read(filename)    
                         
    #Creating arrays for the sections in the configfile 
    keys=config.options('eqparam')  
    values=[]      
    for j in range(len(keys)):        
        values.append(eval(config['eqparam'][keys[j]]))
    eqparamd=dict(zip(keys,values))
    
    keys=config.options('rk4input')  
    values=[]      
    for j in range(len(keys)):        
        values.append(eval(config['rk4input'][keys[j]]))
    rk4inputd=dict(zip(keys,values))

    keys=config.options('initials')  
    values=[]      
    for j in range(len(keys)):        
        values.append(eval(config['initials'][keys[j]]))
    initialsd=dict(zip(keys,values))

    #Creating a dictionary of functions included 
    funclistd={}
    funcparamd={}
    i=0
    while 'func'+str(i) in config:
        funclistd['func'+str(i)]=eval(config['func'+str(i)]['func'])

        keys=config.options('func'+str(i))[1:]  
        values=[]      
        for j in range(len(keys)):        
            values.append(eval(config['func'+str(i)][keys[j]]))
        funcparamd['func'+str(i)]=dict(zip(keys,values))
        i+=1
    
    #packing the function components into one dictionary
    funccompd={'funclist':funclistd,'funcparam':funcparamd} 
    #converting to list-type to allow indexing
    funccomp=dict_to_list(funccompd)     
    eqparam=dict_to_list(eqparamd)     
    rk4input=dict_to_list(rk4inputd)
    initials=dict_to_list(initialsd)     
    #creating output array
    configa=[eqparam, rk4input, funccomp, initials]
    configad=[eqparamd, rk4inputd, funccompd, initialsd]
    #creating array with the names of configa
    configdicnames=lna(['eqparam','rk4input','funccomp','initials'])
    configdic=dict(zip(configdicnames,configad))
    
    #Importing the Variables and initial conditions    
    #Variable_importer()

    #returning the arrays with all needed system parameters and variables
    #return configa, configdic
    return configdic

def dict_to_list(dic):
    dic_to_list=list(dic.values())
    to_list=dic_to_list
    #print(to_list)
    
    i=0
    while type(to_list[i]) == dict:
        to_list[i]=list(to_list[i].values())
        j=0
        while type(to_list[i][j])==dict:
            to_list[i][j]=list(to_list[i][j].values())
            k=0
            while type(to_list[i][j][k])==dict:
                to_list[i][j][k]=list(to_list[i][j][k].values())
                if k<len(to_list[i][j])-1:
                    k+=1
            if j<len(to_list[i])-1:
                j+=1
                
        if i<len(to_list)-1:
            i+=1
    return(to_list)

###Function to import parameters for the Sellers-EBM from a configfile
###with arrays of values
def parameterimporter(filename):
    
    #Importing the configfile
    paras=configparser.ConfigParser()
    paras.read('Config/Parameters/'+filename+'.ini')
    
    #Creating and filling arrays with values for latitudinal belts
    belt=paras.options('belt')
    for i in range(int(len(paras.options('belt')))):
        belt[i]=eval(paras['belt'][belt[i]])
    
    #Creating and filling arrays with values for latitudinal circles
    circle=paras.options('circle')
    for i in range(int(len(paras.options('circle')))):
        circle[i]=eval(paras['circle'][circle[i]])
    
    #Splitting the arrays to get the arrays for northern and southern hemisphere splitted
    beltn,belts=belt[:int(len(belt)/2)],belt[int(len(belt)/2):]
    circlen,circles=circle[:int(len(circle)/2)],circle[int(len(circle)/2):]
    
    #Recombining arrays. Needed because the arrays belt and circle are not
    #ordered from -90° to 90°
    beltcomb=[0]*len(beltn)
    circlecomb=[0]*len(circlen)
    for i in range(len(beltn)):
        belts[i].reverse()
        beltcomb[i]=belts[i]+beltn[i]
    for i in range(len(circlen)):
        circles[i].reverse()
        circlecomb[i]=circles[i]+circlen[i]
    
    #Returning the ordered arrays with systemparameters for latitudinal circles and belts
    return circlecomb, beltcomb


###Function to interpolate the parameterizations given from sellers, into an interpolated
####output with higher resolution
def parameterinterpolator(filename):
    
    #Importing parameters 
    inputparas=parameterimporter(filename)
    
    #defining new latitudinal arrays
    if latitude_NS==True:    
        Latrange=180
        latnewc=np.linspace(-90,90,int(Latrange/latitude_stepsize+1))
        latnewb=np.linspace(-90,90-latitude_stepsize,                               int(Latrange/latitude_stepsize))+latitude_stepsize/2
        latnewb=np.insert(latnewb,0,-90)
        latnewb=np.append(latnewb,90)
    else:
        Latrange=90     
        latnewc=np.linspace(0,90-latitude_stepsize,                                int(Latrange/latitude_stepsize))
        latnewb=np.linspace(0,90-latitude_stepsize,                                int(Latrange/latitude_stepsize))+latitude_stepsize/2
        latnewb=np.insert(latnewb,0,0)
        latnewb=np.append(latnewb,90)
        
    #Interpolation of circle parameters
    newcircle=[0]*len(inputparas[0])
    lat10c=np.linspace(-80,80,17)
    for i in range(len(inputparas[0])):
        zc=np.polyfit(lat10c,inputparas[0][i],10)
        fc=np.poly1d(zc)
        newcircle[i]=fc(latnewc)
        newcircle[i]=newcircle[i][1:]
        newcircle[i]=newcircle[i][:-1]

    #Interpolation of belt parameters
    newbelt=[0]*len(inputparas[1])
    lat10b=np.linspace(-85,85,18)
    for i in range(len(inputparas[1])):
        zb=np.polyfit(lat10b,inputparas[1][i],10)
        fb=np.poly1d(zb)
        newbelt[i]=fb(latnewb)
        newbelt[i]=newbelt[i][1:]
        newbelt[i]=newbelt[i][:-1]
    
    return newcircle, newbelt


###Function to interpolate the parameterizations given from sellers, into an interpolated
####output with higher resolution with stepwise interpolation and averaging
def parameterinterpolatorstepwise(filename):
    
    #Importing parameters 
    inputparas=parameterimporter(filename)

    #Defining new latitudinal arrays
    if latitude_NS==True:    
        Latrange=180
        latnewc=np.linspace(-90+latitude_stepsize,                            90-latitude_stepsize,int(Latrange/latitude_stepsize-1))
        latnewb=np.linspace(-90,90-latitude_stepsize,                            int(Latrange/latitude_stepsize))+latitude_stepsize/2
            
    else:
        Latrange=90     
        latnewc=np.linspace(0,90-latitude_stepsize,                                int(Latrange/latitude_stepsize))
        latnewb=np.linspace(0,90-latitude_stepsize,                                int(Latrange/latitude_stepsize))+latitude_stepsize/2
        latnewb=np.insert(latnewb,0,0)
        latnewb=np.append(latnewb,90)

    ###Interpolation of circle parameters###
    
    #Array for output
    newcircle=np.array([[0]*len(latnewc)]*len(inputparas[0]),dtype=float)
    #Array with 10° latitudal dependence
    lat10c=np.linspace(-80,80,17)
    
    #Loop over each circleparameter
    for k in range(len(inputparas[0])):
        #Loop over the length of latitudes (2nd element to one before last element)
        for i in range(1,len(lat10c)-1):
            #Do a polyfit 2nd order over the element +-1 element (left and right)
            zc=np.polyfit(lat10c[i-1:(i+2)],inputparas[0][k][i-1:(i+2)],2)
            fc=np.poly1d(zc)
            
            #Endpoints are set to the value of the inputparameter (no polyfit here)
            if i == 1:
                for l in range(int(10/latitude_stepsize)):
                    newcircle[k][l]=inputparas[0][k][0]
            if i == (len(lat10c)-2):
                for l in range(int((i+2)*10/latitude_stepsize-1),int((i+3)*10/latitude_stepsize-1)):
                    newcircle[k][l]=inputparas[0][k][-1]
            
            #Write for the new latitudal dependence the corresponding values 
            #from the polyfit into the given outputarray
            #Loop over the element to the right of i-1 to the element to left of i+1
            for j in range(int(2*10/latitude_stepsize-1)):  
                if newcircle[k][int(i*10/latitude_stepsize+j)] == 0:
                    newcircle[k][int((i)*10/latitude_stepsize+j)]=                    fc(latnewc[int((i)*10/latitude_stepsize+j)])
                
                #The inbetween value are calculated twice (from left and right) which are now
                #averaged to give a smooth new parameterization
                else:
                    newcircle[k][int((i)*10/latitude_stepsize+j)]=np.mean([fc(latnewc                    [int((i)*10/latitude_stepsize+j)]),newcircle[k][int((i)*10/latitude_stepsize+j)]])
                
                   
    ###Interpolation of belt parameters###
    
    #Array for output
    newbelt=np.array([[0]*len(latnewb)]*len(inputparas[1]),dtype=float)
    #Array with 10° latitudal dependence
    lat10b=np.linspace(-85,85,18)  
    
    #Loop over each beltparameter
    for k in range(len(inputparas[1])):
        #Loop over the length of latitudes (2nd element to one before last element)
        for i in range(1,len(lat10b)-1):
            #Do a polyfit 2nd order over the element +-1 element (left and right)
            zb=np.polyfit(lat10b[i-1:(i+2)],inputparas[1][k][i-1:(i+2)],2)
            fb=np.poly1d(zb)
            
            #Endpoints are set to the value of the inputparameter (no polyfit here)
            if i == 1:
                for l in range(int(0.5*10/latitude_stepsize)):
                    newbelt[k][l]=inputparas[1][k][0]
            if i == (len(lat10b)-2):
                for l in range(int((i+0.5)*10/latitude_stepsize),len(newbelt[k])):
                    newbelt[k][l]=inputparas[1][k][-1]
            
            #Write for the new latitudal dependence the corresponding values 
            #from the polyfit into the given outputarray
            #Loop over the element to the right of i-1 to the element to left of i+1
            for j in range(int(2*10/latitude_stepsize)):  
                if newbelt[k][int((i-0.5)*10/latitude_stepsize+j)] == 0:
                    newbelt[k][int((i-0.5)*10/latitude_stepsize+j)]=                    fb(latnewb[int((i-0.5)*10/latitude_stepsize+j)])
                
                #The inbetween value are calculated twice (from left and right) which are now
                #averaged to give a smooth new parameterization
                else:
                    newbelt[k][int((i-0.5)*10/latitude_stepsize+j)]=np.mean([fb(latnewb                    [int((i-0.5)*10/latitude_stepsize+j)]),newbelt[k][int((i-0.5)*10/latitude_stepsize+j)]])
                    
    return newcircle, newbelt


#Function to rewrite parameters with arrays of parameters from the Sellers parameterinterpolator
#Hardcoded! So take care on which index the sellers functions are placed, standard is:
#func0 = Incoming Radiation , func1 = Outgoing Radiation, func2 = Transfer, ...
def add_sellersparameters(config,importer,file,solar,albedo): 
    #solar and albedo are to be set to True or False 
    
    #importing the new parameter arrays
    paras=[[dp,dz,K_h,K_wv,K_o,a],[b,Z,Q]]=importer(file)
    
    #rewriting the transfer parameters with conversion to SI units
    config[2][1][2][2]=lna(K_wv)*10**5
    config[2][1][2][3]=lna(K_h)*10**6
    config[2][1][2][4]=lna(K_o)*10**2
    config[2][1][2][6]=lna(a)/100
    config[2][1][2][13]=lna(dp)
    config[2][1][2][15]=lna(dz)*1000
    
    #rewriting the incoming radiation parameters with conversion + choice to be activated or not
    if albedo==True:
        config[2][1][0][5][0]=lna(Z)
        config[2][1][0][5][1]=lna(b)
    if solar==True:
        Vars.Solar=lna(Q)*1.327624658#1.2971#
        
    return config, paras
    
