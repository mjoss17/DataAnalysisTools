"""
Meta_Model.py
Author: Matt Joss

"""
import time
import Single_Models as a
import PreProcess_Avila as av 
import PreProcess_Glass as gl 
import PreProcess_Iris as ir
import GA_ForParams as ga
from matplotlib.backends.backend_pdf import PdfPages

log=[]

def get_data():
    return av.get_split_data()

def compareTechniques():
    train_X, train_y, test_X, test_y = get_data()
    getAccuracies(train_X, train_y, test_X, test_y)

def getAccuracies(train_X, train_y, test_X, test_y):
    with PdfPages("FILLER.pdf") as pdf_pages:
        title = "OG"

        start = time.process_time()
        dt, dt_loss = ga.bigBang("dt", 100, 20, pdf_pages=pdf_pages)
        dt_time = time.process_time() - start
        start = time.process_time()
        rf, rf_loss = ga.bigBang("rf", 49, 20, pdf_pages=pdf_pages)
        rf_time = time.process_time() - start
        start = time.process_time()
        lr, lr_loss = a.logisticRegression(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", Log Regression"))
        lr_time = time.process_time() - start
        start = time.process_time()
        kn, kn_loss = ga.bigBang("kn", 49, 20, pdf_pages=pdf_pages)
        kn_time = time.process_time() - start
        start = time.process_time()
        sv, sv_loss = a.sVM(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", SVN"))
        sv_time = time.process_time() - start
        start = time.process_time()
        nn, nn_loss = ga.bigBang("nn", 25, 35, pdf_pages=pdf_pages)
        nn_time = time.process_time() - start

        log.append("DT:  Acc = %0.5f   Loss = %0.5f  Time = %0.5f"% (dt, dt_loss, dt_time))
        log.append("RF:  Acc = %0.5f   Loss = %0.5f  Time = %0.5f"% (rf, rf_loss, rf_time))
        log.append("LR:  Acc = %0.5f   Loss = %0.5f  Time = %0.5f"% (lr, lr_loss, lr_time))
        log.append("KN:  Acc = %0.5f   Loss = %0.5f  Time = %0.5f"% (kn, kn_loss, kn_time))
        log.append("SV:  Acc = %0.5f   Loss = %0.5f  Time = %0.5f"% (sv, sv_loss, sv_time))
        log.append("NN:  Acc = %0.5f   Loss = %0.5f  Time = %f"% (nn, nn_loss, nn_time))
        

        print('\n'.join(log))

compareTechniques()