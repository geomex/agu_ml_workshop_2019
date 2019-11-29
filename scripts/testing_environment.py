try:
    from os.path import abspath
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.signal import medfilt2d
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    
    print('\nCongrats! You are ready to run through the Tutorial!\n')
except:
    print("hmm...somthething isn't quite right :-(. try to recreate your environment. \n\nif that still doesn't attempt to reclone the repository and recreate the environment. \n\nif that still doesn't work, shoot me an email and i can try my best to assist you otherwise you can wait until the workshop to get help, ideally before we start the lesson. \n\nif you are a procrastinator and running this for the first time at the workshop, raise your hand for assistance and someone will come around to assist you. :-) cheers!")