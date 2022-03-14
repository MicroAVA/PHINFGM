# coding: utf-8
# All needed packages
import argparse
import pandas as pd
import math as math
import numpy as np
import csv
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier
import sklearn.svm as svm

# Import my files
from load_datasets import *
from pathScores_functions import *
from get_Embeddings_FV import *
from training_functions import *
from GIP import *
from snf_code import *
import os
import warnings
warnings.filterwarnings("ignore")


######################################## START MAIN #########################################
#############################################################################################
def main():
    # get the parameters from the user
    args = parse_args()
    ## get the start time to report the running time
    t1 = time.time()

    ### Load the input data - return all pairs(X) and its labels (Y)..
    allH, allV, allHsim, allVsim, HrVr, R, X, Y = load_datasets(args.data)

    # create 2 dictionaries for host virus. the keys are their order numbers
    hostID = dict([(h, i) for i, h in enumerate(allH)])
    virusID = dict([(v, i) for i, v in enumerate(allV)])
    #-----------------------------------------
    ###### Define different classifiers

    # 1-Random Forest
    rf = RandomForestClassifier(n_estimators=200, n_jobs=10, random_state=55, class_weight='balanced', criterion='gini')

    # 2-Neural Network
    NN = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)

    # 3-Adaboost classifier
    ab = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=2,
                                                   min_samples_leaf=1, max_features='auto', random_state=10,
                                                   max_leaf_nodes=None,
                                                   class_weight='balanced'), algorithm="SAMME", n_estimators=90,
                            random_state=32)

    # 4-Xgboost classifier
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27,
                        use_label_encoder=False,
                        eval_metric='mlogloss')
    # 5-SVM
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)

    #________________________________________________________________
    # 10-folds Cross Validation...............
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 22)
    skf.get_n_splits(X, Y)
    foldCounter = 1
    # all evaluation lists
    correct_classified = []
    ps = []
    recall = []
    roc_auc = []
    average_precision = []
    f1 = []
    Pre = []
    Rec = []
    AUPR_TEST = []
    TN = []
    FP = []
    FN = []
    TP = []
    all_hv_PredictedScore = []

    #Create file to write the novel interactions based on predicted scores
    novel_HV_file = 'Novel_VHIs/'+args.data+'/'+args.data+'_top_novel_VHIs.csv'
    if not os.path.exists(novel_HV_file):
        file = open(novel_HV_file, 'w')
        file.close()
    # Start training and testing
    for train_index, test_index in  skf.split(X,Y):

        print("*** Working with Fold %i :***" %foldCounter)
        
        #first thing with R train to remove all edges in test (use it when finding path)
        train_HV_Matrix = Mask_test_index(test_index, X, Y, HrVr, hostID, virusID)
        HrVr_train = train_HV_Matrix.transpose()


        HHsim = []
        VVsim = []

        for sim in allHsim:
            HHsim.append(sim)


        for sim in allVsim:
            VVsim.append(sim)


        fused_simHr = SNF(HHsim,K=5,t=3,alpha=1.0)
        fused_simVr = SNF(VVsim,K=5,t=3,alpha=1.0)
        ##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # insert node2vec code here to generate embedding in the same code.....
        #------------------------------ node2vec ------------------------------

        virusFV, hostFV = get_FV_host_virus(foldCounter, allV, allH, args.data)
        
        # Calculate cosine similarity for each host pair, and for each virus pair
        cos_simHH = Cosine_Similarity(hostFV)
        cos_simVV = Cosine_Similarity(virusFV)
        # normalize simiarities to be in positive range [0,1]
        cos_simHH = normalizedMatrix(cos_simHH)
        cos_simVV  = normalizedMatrix(cos_simVV)
        #--------------------------------------------------------------------- 

        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G1)
        sumHHH, maxHHH = HHH_VVV_sim(fused_simHr)
        sumVVV, maxVVV = HHH_VVV_sim(fused_simVr)
        
        sumHHV,maxHHV = metaPath_Hsim_HV(fused_simHr,HrVr_train,2)
        sumHVV,maxHVV = metaPath_HV_Vsim(fused_simVr,HrVr_train,2)

        sumHHHV,_= metaPath_Hsim_HV(sumHHH,HrVr_train,3)
        _,maxHHHV = metaPath_Hsim_HV(maxHHH,HrVr_train,3)

        sumHVVV,_ = metaPath_HV_Vsim(sumVVV,HrVr_train,3)
        _,maxHVVV = metaPath_HV_Vsim(maxVVV,HrVr_train,3)

        sumHVHV,maxHVHV = metaPath_HVHV(HrVr_train)
        sumHHVV,maxHHVV = metaPath_HHVV(HrVr_train,fused_simHr,fused_simVr)
    #============================================================================== 
        # Generate all featres from the matrix multiplication of each path strucutre
        # list for each feature (Graph G2)
        sumHHH2, maxHHH2 = HHH_VVV_sim(cos_simHH)
        sumVVV2, maxVVV2 = HHH_VVV_sim(cos_simVV)
        
        sumHHV2,maxHHV2 = metaPath_Hsim_HV(cos_simHH,HrVr_train,2)
        sumHVV2,maxHVV2 = metaPath_HV_Vsim(cos_simVV,HrVr_train,2)

        sumHHHV2,_ = metaPath_Hsim_HV(sumHHH2,HrVr_train,3)
        _,maxHHHV2 = metaPath_Hsim_HV(maxHHH2,HrVr_train,3)

        sumHVVV2,_ = metaPath_HV_Vsim(sumVVV2,HrVr_train,3)
        _,maxHVVV2 = metaPath_HV_Vsim(maxVVV2,HrVr_train,3)

        sumHVHV2,maxHVHV2 = metaPath_HVHV(HrVr_train)
        sumHHVV2,maxHHVV2 = metaPath_HHVV(HrVr_train,cos_simHH,cos_simVV)

    #==============================================================================  
    ### Build feature vector and class labels
        HV_score = []
        for i in range(len(allH)):
            for j in range(len(allV)):
                pair_scores = (allH[i], allV[j],\
                            # path scores from G1
                               sumHHV[i][j],sumHHHV[i][j],\
                               sumHVV[i][j],sumHVVV[i][j], sumHHVV[i][j], sumHVHV[i][j],\
                               maxHHV[i][j],maxHHHV[i][j], \
                               maxHVV[i][j],maxHVVV[i][j],maxHHVV[i][j],maxHVHV[i][j],\
                            # path scores from G2
                               sumHHV2[i][j],sumHHHV2[i][j],\
                               sumHVV2[i][j],sumHVVV2[i][j], sumHHVV2[i][j], sumHVHV2[i][j],\
                               maxHHV2[i][j],maxHHHV2[i][j], \
                               maxHVV2[i][j],maxHVVV2[i][j],maxHHVV2[i][j],maxHVHV2[i][j])
                HV_score.append(pair_scores)
        
        features = []
        class_labels = []
        HV_pair = []
        # Build the feature vector - Concatenate features from G1,G2
        for i in range(len(HV_score)):
            hr = HV_score[i][0]
            vr = HV_score[i][1]
            edgeScore = HV_score[i][2], HV_score[i][3], HV_score[i][4],HV_score[i][5],\
                        HV_score[i][8],HV_score[i][9], HV_score[i][10], HV_score[i][11],\
                        HV_score[i][14], HV_score[i][15],HV_score[i][16],HV_score[i][17],HV_score[i][18],\
                        HV_score[i][20], HV_score[i][21],HV_score[i][22]
           
            hv = HV_score[i][0], HV_score[i][1]
            HV_pair.append(hv)
            features.append(edgeScore)
            # same label as the begining
            label = R[hr][vr]
            class_labels.append(label)

        ## Start Classification Task
        # featureVector and labels for each pair
        XX = np.asarray(features)
        YY = np.array(class_labels)

        #Apply normalization using MaxAbsolute normlization
        max_abs_scaler = MaxAbsScaler()
        X_train = max_abs_scaler.fit(XX[train_index]) 
        X_train_transform = X_train.transform(XX[train_index])

        X_test_transform = max_abs_scaler.transform(XX[test_index])

        # Apply different oversampling techniques:
        ros = RandomOverSampler(random_state=10)
        # sm = SMOTE(random_state=10)
        # ada = ADASYN(random_state=10)
        # X_res, y_res= ros.fit_sample(X_train_transform, YY[train_index])
        X_res, y_res = ros.fit_resample(X_train_transform, YY[train_index])

        # fit the model
        ab.fit(X_res, y_res)
        predictedClass = ab.predict(X_test_transform)
        predictedScore = ab.predict_proba(X_test_transform)[:, 1]

        #Find the novel interactions based on predicted scores
        fold_hv_score = []
        for idx, c in zip(test_index,range(0,len(predictedScore))):
            # write host, virus, predicted score of class1, predicted class, actual class
            hvSCORE = str(HV_pair[idx]),predictedScore[c],predictedClass[c],YY[idx]
            all_hv_PredictedScore.append(hvSCORE)

        # ------------------- Print Evaluation metrics for each fold --------------------------------
        print("@@ Validation and evaluation of fold %i @@" %foldCounter)
        print(YY[test_index].shape, predictedClass.shape)

        cm = confusion_matrix(YY[test_index], predictedClass)
        TN.append(cm[0][0])
        FP.append(cm[0][1])
        FN.append(cm[1][0])
        TP.append(cm[1][1])
        print("Confusion Matrix for this fold")
        print(cm)

        print("Correctly Classified Instances: %d" %accuracy_score(Y[test_index], predictedClass, normalize=False))
        correct_classified.append(accuracy_score(Y[test_index], predictedClass, normalize=False))

        #print("Precision Score: %f" %precision_score(Y[test_index], predictedClass))
        ps.append(precision_score(Y[test_index], predictedClass,average='weighted'))

        #print("Recall Score: %f" %recall_score(Y[test_index], predictedClass)
        recall.append(recall_score(Y[test_index], predictedClass, average='weighted'))

        print("F1 Score: %.6f" %f1_score(Y[test_index], predictedClass, average='weighted'))
        f1.append(f1_score(Y[test_index], predictedClass,average='weighted'))

        print("Area ROC: %.6f" %roc_auc_score(Y[test_index], predictedScore))
        roc_auc.append(roc_auc_score(Y[test_index], predictedScore))

        p, r, _ = precision_recall_curve(Y[test_index],predictedScore,pos_label=1)
        aupr = auc(r, p)
        print("AUPR auc(r,p) = %.6f" %aupr)
        AUPR_TEST.append(aupr)

        Pre.append(p.mean())
        Rec.append(r.mean())
        average_precision.append(average_precision_score(Y[test_index], predictedScore))

        print(classification_report(Y[test_index], predictedClass))
        print('--------------------------------------------------')
        foldCounter += 1
        #,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

    # Write predicted scores into file to find novel interactions:
    hv_df = pd.DataFrame(all_hv_PredictedScore, columns=['HV_pair', 'Predicted_score_class1', 'Predicted_Class', 'Actual_Class'])
    hv_df = hv_df.sort_values(by='Predicted_score_class1', ascending=False)
    
    hv_df = hv_df[hv_df['Predicted_Class']==1]
    novel_hv = hv_df[hv_df['Actual_Class']==0]

    novel_hv.to_csv(novel_HV_file,sep='\t', index=None)
    #--------------------------------------------------------------------
    ############# Evaluation Metrics ####################################
    # Confusion matrix for all folds
    ConfMx = np.zeros((cm.shape[0],cm.shape[0]))
    ConfMx[0][0] = str( np.array(TN).sum() )
    ConfMx[0][1] = str( np.array(FP).sum() )
    ConfMx[1][0] = str( np.array(FN).sum() )
    ConfMx[1][1] = str( np.array(TP).sum() )

    ### Print Evaluation Metrics.......................
    print("Result(Correct_classified): " + str( np.array(correct_classified).sum() ))
    print("Results:precision_score = " + str( np.array(ps).mean().round(decimals=8) ))
    print("Results:recall_score = " + str( np.array(recall).mean().round(decimals=8) ))
    print("Results:f1 = " + str( np.array(f1).mean().round(decimals=8) ))
    print("Results:roc_auc = " + str( np.array(roc_auc).mean().round(decimals=8) ))
    print("Results: AUPR on Testing auc(r,p) = " + str( np.array(AUPR_TEST).mean().round(decimals=8)))
    print("Confusion matrix for all folds")
    print(ConfMx) 
    print('_____________________________________________________________')
    print('Running Time for the whole code:', time.time()-t1)  
    print('_____________________________________________________________')  
#####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    main()
#####-------------------------------------------------------------------------------------------------------------
####################### END OF THE CODE ##########################################################################
