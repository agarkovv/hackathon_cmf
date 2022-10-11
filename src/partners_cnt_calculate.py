import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')
from datetime import tzinfo, timedelta, datetime, date

orders = pd.read_csv("../data/origin/orders.csv")
partners = pd.read_csv("../data/origin/partners_delays.csv")

partners = partners.rename(columns={"dttm": "date"})
orders['date'] = pd.to_datetime(orders['date'])
partners['date'] = pd.to_datetime(partners['date'])


def split_hour(d):
    year, month, day, hour = d.timetuple()[0:4]
    return d.date(), hour


index_o = pd.MultiIndex.from_tuples([(del_id, date, hour) for del_id, (date, hour)
                                     in zip(orders["delivery_area_id"], orders["date"].apply(split_hour))],
                                    names=('delivery_area_id', 'date', 'hour'))

index_p = pd.MultiIndex.from_tuples([(del_id, date, hour) for del_id, (date, hour)
                                     in zip(partners["delivery_area_id"], partners["date"].apply(split_hour))],
                                    names=('delivery_area_id', 'date', 'hour'))

v_orders = np.array(orders['orders_cnt'])
v_partners = np.array(partners['partners_cnt'])
v_delay_rate = np.array(partners['delay_rate'])
orders = pd.DataFrame(v_orders, index=index_o, columns=['orders_cnt'])
partners = pd.DataFrame(np.array([v_partners, v_delay_rate]).T, index=index_p, columns=['partners_cnt', 'delay_rate'])

df = orders.merge(partners, left_index=True, right_index=True, how='outer')
df.insert(loc=0, column='inv_perc', value=np.array(df["orders_cnt"]) / np.array(df["partners_cnt"]))
df = df.fillna(0)
df["delay_rate"] = df["delay_rate"].apply(lambda d: 1 if d < 0.05 else 0)
df = df.rename(columns={"delay_rate": "delay_rate<0.05?"})
df.insert(loc=2, value=[p ** 2 for p in df["partners_cnt"]], column='partners^2')

'''
X = df[["inv_perc", "orders_cnt", "partners^2", "partners_cnt"]]
Y = df["delay_rate<0.05?"]
model = LinearSVC(C=0.01)
model.fit(X, Y)
print("coef=", model.coef_[0], "intercept=", model.intercept_[0])
'''

# coef= [-0.09744588 -0.12323587  0.01293844 -0.05533381] intercept= 1.3146978398172149

u = np.array([-0.09744588, -0.12323587, 0.01293844, -0.05533381])
c = 1.3146978398172149


def f(orders, partners):
    x = np.array([orders / partners, orders, partners ** 2, partners])
    return x.T @ u + c > 3 / 4


def get_min_partner(orders):
    if orders == 0:
        return 0
    else:
        partners = 1
        while not f(orders, partners):
            partners += 1
        return partners


'''
for o in range(25):
    print(o, get_min_partner(o))
'''

ans0 = pd.read_csv("../data/orders_cnt_prediction/ans0.csv")
ans1 = pd.read_csv("../data/orders_cnt_prediction/ans1.csv")
ans2 = pd.read_csv("../data/orders_cnt_prediction/ans2.csv")

ids = ['id__{}'.format(i) for i in range(593)]
df_ans = pd.concat([ans0, ans1, ans2])
for i, id in enumerate(ids):
    df_ans[id] = df_ans[id].apply(lambda x: x * i)
df_delivery_area_id = df_ans[ids].sum(axis=1)
df_ans.insert(loc=0, value=df_delivery_area_id, column='delivery_area_id')
df_ans = df_ans.drop(columns=ids)
df_ans = df_ans.drop(columns=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
df_ans['date'] = pd.to_datetime(df_ans['date'])
index_ans = pd.MultiIndex.from_tuples([(del_id, date, hour) for del_id, (date, hour)
                                       in zip(df_ans["delivery_area_id"], df_ans["date"].apply(split_hour))],
                                      names=('delivery_area_id', 'date', 'hour'))
df_ans_ = pd.DataFrame([o for o in df_ans["orders_cnt"]], index=index_ans, columns=['orders_cnt'])
df_ans_.insert(loc=0, value=df_ans_["orders_cnt"].apply(get_min_partner), column='partners_cnt')
df_ans_.reset_index().to_csv('../data/partners_cnt/partners_and_orders.csv')
