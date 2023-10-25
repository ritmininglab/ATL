
import numpy as np
def checkConverge(res,ind,thres=0.005):
    if(np.abs(res[ind-1]-res[ind-2])<thres):
        return True
    else:
        return False
    

    
    
    
    