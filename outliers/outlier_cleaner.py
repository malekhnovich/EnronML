#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = {}
    good_list  = []


    for i in range(len(predictions)):
        error = net_worths[i]-predictions[i]
        if error<0:
            error = error*-1
        errors[i] = error

    maxErrors = sorted(errors.values())

    #getting the last 10 elements
    maxErrors= maxErrors[-9:]


    index = 0
    for e in errors.values():
        if e not in maxErrors:
            cleaned_data.append((int(ages[index]),int(net_worths[index]),int(errors[index])))
        index += 1


    return cleaned_data




    


