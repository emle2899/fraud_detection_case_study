import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

import matplotlib.pyplot as plt
import seaborn as sns

def randomForest(X_train,y_train):

    rf = RandomForestClassifier(n_estimators=21, oob_score=True, random_state=42)

    rf.fit(X_train, y_train)

    return rf

def heatplot(X):
    '''
    Plot correlation heatmap
    INPUT:
    matrix = DataFrame
    OUTPUT:
    sns.heatmap
    '''
    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(X, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(X, mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title('Pearson Correlation Coeficient Matrix')
    plt.tight_layout()
    plt.savefig('images/corr_plot')
#     plt.show()

def identify_fraud(item):
    if item.find('fraud') == 0:
        return 1
    else:
        return 0

def plot_roc(X, y, clf_class, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    clf = clf_class(**kwargs)
    clf.fit(X_train, y_train)

    clf_class = pickle.dumps(clf)
    # Predict probabilities, not classes
    y_prob = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label=clf.__class__.__name__ + f' {round(roc_auc, 2)}')


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')

def plot_roc_curve(X,y,title):
    plot_roc(X, y, GradientBoostingClassifier)
    plot_roc(X, y, AdaBoostClassifier)
    plot_roc(X, y, RandomForestClassifier)
    plot_roc(X, y, LogisticRegression)
    plot_roc(X, y, DecisionTreeClassifier)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.legend(loc="lower right")
#     plt.savefig('images/'+title+'.png')
    plt.show()

if __name__ == '__main__':
    df = pd.read_json('data/data.json')
    print(df.head())

    df = pd.read_json('data/data.json')
    df['Isfraud'] = df['acct_type'].apply(identify_fraud)

    df_re = df.drop(['has_header','venue_country','venue_latitude', 'venue_longitude', 'venue_name', 'venue_state'],axis=1)
    df_final = df_re.dropna()

    # lb_make = LabelEncoder()
    # df_final["country_code"] = lb_make.fit_transform(df_final["country"])
    # df_final["payout_type_code"] = lb_make.fit_transform(df_final["payout_type"])

    df_final = df_final.drop(['gts','num_order', 'num_payouts', 'sale_duration2','approx_payout_date','country','payout_type','acct_type','currency','description','email_domain','listed','name','org_desc','org_name','payee_name','previous_payouts','ticket_types','venue_address'],axis=1)

    # Separate majority and minority classes
    df_majority = df_final[df_re.Isfraud==0]
    df_minority = df_final[df_re.Isfraud==1]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                     replace=False,    # sample without replacement
                                     n_samples=df_final['Isfraud'].sum(),     # to match minority class
                                     random_state=123) # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    y_down = df_downsampled.pop('Isfraud')

    plot_roc_curve(df_downsampled,y_down,'roc_curve')

    X_train, X_test, y_train, y_test = train_test_split(df_downsampled, y_down, random_state=42)
    correlated = X_train.corr(method='pearson', min_periods=1)
    heatplot(correlated)

    rf = randomForest(X_train,y_train)
    predicted = rf.predict(X_test)
    prob = rf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predicted)
    class_report = classification_report(y_test,predicted)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)
    y_predict = clf.predict(X_test)
    class_report2 = classification_report(y_test,y_predict)

    # save the model to disk
    with open ('rf_model.pkl','wb') as f:
        pickle.dump(rf, f)

    with open ('gb_model.pkl','wb') as f:
        pickle.dump(clf, f)

    with open ('gb_classification.txt','wb') as f:
        pickle.dump(class_report2, f)

    with open ('rf_classification.txt','wb') as f:
        pickle.dump(class_report, f)
