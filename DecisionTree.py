"""

Created by:

Sayan Nanda (snanda@usc.edu)

&

Srihari Akash Chinam (chinam@usc.edu)

"""

#Reading text data provided
txt_object = open("dt-data.txt","r")
txt_data = txt_object.read()

attribute_list = txt_data.split(')')[0][1:].split(',')
attribute_list = [attribute.strip(' ') for attribute in attribute_list]

#Converting the data read into a dataframe to make it more readable and accessible
import pandas
data_frame = pandas.DataFrame(columns = attribute_list)

for row in txt_data.split('\n')[2:]:
    if row == '':
        break
    row = [item.strip(' ;')  for item in (row.split(':')[1]).split(',')]
    data_frame.loc[len(data_frame)] = row 
target_attribute =  'Enjoy'    


#Calculates entropy of the current node using target attribute
def entropyOfSet(data_frame, target_attribute):
    entropy = 0
    len_df = len(data_frame)
    from math import log
    for value in data_frame[target_attribute].value_counts():
        entropy = entropy + (float(value)/len_df)*(log((float(len_df)/value),2))
    return entropy 

 
#calculates information gain between current node and split attribute    
def informationGain(data_frame, entropy_set, target_attribute, attribute):
    main_entropy = entropy_set
    local_entropy = 0
    len_df = len(data_frame)
    for value in data_frame[attribute].unique():
        sub_data = data_frame[data_frame[attribute] == value]
        local_entropy += ((float(len(sub_data))/len_df) * entropyOfSet(sub_data,target_attribute))
    info_gain = main_entropy-local_entropy
    return info_gain

#Reolves situations for decision tree generation
def resolver(data_frame, target_attribute):
    if len(data_frame[data_frame[target_attribute] == 'No']) > len(data_frame[data_frame[target_attribute] == 'Yes']):
            return 'No'
        
    elif len(data_frame[data_frame[target_attribute] == 'No']) < len(data_frame[data_frame[target_attribute] == 'Yes']):
            return 'Yes'
        
    else:
            return 'Tie'

#Generates decision tree
def decisionTree(data_frame, target_attribute):
    new_list = []
    total_len = len(data_frame)
    if len(data_frame[data_frame[target_attribute] == 'Yes']) == total_len:
        #print 'Yes'
        new_list.append('Yes')
    elif len(data_frame[data_frame[target_attribute] == 'No']) == total_len:
        #print 'No'
        new_list.append('No')
    elif len(data_frame.columns) == 1:
        new_list.append(resolver(data_frame, target_attribute))
    else:
        max_info_gain = 0
        eos = entropyOfSet(data_frame, target_attribute)
        split_attribute =''
        for attribute in data_frame.columns:
            if attribute == target_attribute:
                continue          
            info_gain = informationGain(data_frame, eos, target_attribute, attribute)
            if info_gain>max_info_gain:
                max_info_gain = info_gain
                split_attribute = attribute
        if(max_info_gain) == 0:
            new_list.append(resolver(data_frame, target_attribute))
        else:    
            #print split_attribute
            new_list.append(split_attribute)
            for value in data_frame[split_attribute].unique():
                #print value
                newer_list = []
                newer_list.append('<'+value+'>')
                newer_list.append(decisionTree(data_frame[data_frame[split_attribute] == value].drop(split_attribute,1),target_attribute))
                new_list.append(newer_list)
    return new_list

#Prints the decision tree in readable format
def Tree_print(dect, l=0):
    print ('    ' * (l-1)),'---' * (l>0),dect[0]
    for i in dect[1:]:
        if type(i) == str:
            print '    '*l,'---',i 
        elif type(i) == list:
            Tree_print(i,l+1)

#Provides Prediction for user input
def predictor(dectree):
    
    if dectree[0]=='Yes':
        print "\nWill Enjoy"
        return 1
    elif dectree[0]=='No':
        print "\nWill not Enjoy"
        return 1
    elif dectree[0]=='Tie':
        print "\nCannot Predict"
        return 1
    
    x = dectree[0]
    user = raw_input("\nEnter value for attribute \"" + x + "\" - ")
    user = '<' + user + '>'
    for i in dectree:
        if user in i:
            predictor(i[1])
            break
    
dt = decisionTree(data_frame, target_attribute)
Tree_print(dt,1)
predictor(dt)
