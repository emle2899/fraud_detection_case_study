import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
import random
import requests
from sqlalchemy import create_engine


class fraud():
    def __init__(self, df, y):
        self.df = df
        self.y = y
        pass
    def event_mask_build(self):
        self.unique_y = np.unique(self.y)
    def mask_replace(self):
        self.u = []
        for i in range(len(self.unique_y)):
            if 'fraudster' in self.unique_y[i]:
                self.u.append(1)
            else:
                self.u.append(0)
        self.u = np.array(self.u).reshape(len(self.u),1)
        self.y.replace(to_replace=self.unique_y, value = self.u, inplace=True)

    def randomForest(x,y):
       X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
       rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)

       rf.fit(X_train, y_train)
       predicted = rf.predict(X_test)
       accuracy = accuracy_score(y_test, predicted)

       print(f'score estimate: {rf.oob_score_:.3}')
       print(f'Mean accuracy score: {accuracy:.3}')

       return X_train, X_test, y_train, y_test
    def get_db():
        """Opens a new database connection if there is none yet for the
        current application context.
        """
        if not hasattr(g, 'sqlite_db'):
            g.sqlite_db = connect_db()
        return g.sqlite_db
    def identify_fraud(item):
        if item.find('fraud') == 0:
            return 1
        else:
            return 0
    def requester (self):
        #api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
        url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
        sequence_number = 0
        response = requests.get(url)
        x = response.json()
        df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in x.items() ]))
        df.fillna(0, inplace=True)
        df = df.drop(['approx_payout_date','num_order','object_id','gts', 'sale_duration2', 'has_header','venue_country','venue_latitude', 'venue_longitude', 'venue_name', 'venue_state'],axis=1)
        lb_make = LabelEncoder()
        self.output = df.drop(['country','payout_type','currency','description','email_domain','listed','name','org_desc','org_name','payee_name','previous_payouts','ticket_types','venue_address'],axis=1)
    def nan(self):
        stop= self.output.isnull().values.any()
        return stop
    def jasmine(self):
        filename = 'gb_model.pkl'
        self.loaded_model = pickle.load(open(filename, 'rb'))
    def pred(self):
        self.predict=self.loaded_model.predict(self.output)
    def pred_proba(self):
        self.pred_prob = self.loaded_model.predict_proba(self.output)
    def update_output_1(self):
        self.output['fraud'] = self.predict
    def update_output_2(self):
        self.output['fraud_prob_0'] = self.pred_prob[0][0]
        self.output['fraud_prob_1'] = self.pred_prob[0][1]
    def write_to_psql(self):
        engine = create_engine('postgresql://rosskantor@localhost:5432/fraud')
        self.output.to_sql('fraud_checker', engine, if_exists='append')


if __name__ =="__main__":

    stopper = 'Good to Go'
    df = pd.read_json('./data/data.json')
    y = df.pop('acct_type')
    f = fraud(df, y)
    f.event_mask_build()
    f.mask_replace ()
    f.requester()
    stop = f.nan()
    if stop == False:
        f.jasmine()
        f.pred()
        f.pred_proba()
        f.update_output_1()
        f.update_output_2()
        f.write_to_psql()
    else:
        stopper = 'Nan Encountered'
