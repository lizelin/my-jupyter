# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# 房产公司预测区域房价中位线


# %%
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# print(HOUSING_URL)
# fetch_housing_data(HOUSING_URL, HOUSING_PATH)


# %%
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data(HOUSING_PATH)
housing.head()

# %% [markdown]
# # 字段含义
# - longitude：经度
# - latitude：纬度
# - housing_median_age: 房龄
# - total_rooms: 房间总数
# - total_bedrooms: 卧室数目
# - population: 人口
# - households: 家庭
# - median_income: 收入中位数
# - median_housing_value: 房价中位数
# - ocean_proximity: 临近？
# 

# %%
housing.info()


# %%
housing["ocean_proximity"].value_counts()


# %%
# 数值属性的摘要
housing.describe()


# %%
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# %%
# 创建测试集
# 该数据集随机，并不完美。最好每次生成的数据集一致（固定）
import numpy as np
def split_train_test(data, test_radio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_radio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), " train +", len(test_set), " test")


# %%
import hashlib
def test_set_check(identifier, test_radio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_radio

def split_train_test_by_id(data, test_radio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_radio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # add index clolumn, 使用index列进行索引
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
test_set.head()

housing_with_id['id'] = housing["longitude"] * 1000 + housing["latitude"]  # 使用自定义的id列进行索引拆分
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
test_set.head()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) # 42是常用的种子
test_set.head()


# %%
housing["median_income"].hist()


# %%
# 收入分类（收入的中位数大致聚集在2-5万左右）
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing["income_cat"].hist()


# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# %%
# 看看所有住房数据 根据 收入类别的比例分布
strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# %%
housing["income_cat"].value_counts()/len(housing)


# %%
# 比较不同抽样中，收入的比例是否一致
def income_cat_proportions(data):
    return data["income_cat"].value_counts()/len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set)
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"]/compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"]/compare_props["Overall"] - 100


# %%
compare_props


# %%
# 恢复数据集
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    

# %% [markdown]
# # 从数据探索和可视化中获得洞见

# %%
housing = strat_train_set.copy()


# %%
# 将地理数据可视化
housing.plot(kind="scatter", x="longitude", y="latitude")


# %%
# 将alpha属性设置为0.1，更好的看到数据分布
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# %%
# s代表人口数量（圆半径），c代表房价（颜色），jet是预置的色模板
housing.plot(kind="scatter", x="longitude", y="latitude", s=housing["population"]/100,label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()


# %%
# 属性间相关系数
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# %%
import matplotlib.image as mpimg
california_img = mpimg.imread("images" + "/california.png")
ax = housing.plot(kind="scatter", x="longitude", y="latitude", s=housing["population"]/100,label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label("Median House Value", fontsize = 16)

plt.legend(fontsize=16)
plt.show()


# %%
# 绘制各个属性间的相关性
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# %%
# 最有影响的是收入中位数
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# %%
# 几个计算属性
# 每个家庭房间数
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# 卧室数 对比 房间数
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# 每个家庭人口
housing["population_per_household"] = housing["population"]/housing["households"]


# %%
corr_matrix = housing.corr()


# %%
corr_matrix["median_house_value"].sort_values(ascending=False)


# %%
# 分离预测数据和标签
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# %%
# 处理缺失数据
# 删除空数据
# housing.dropna(subset=["total_bedrooms"])
# housing.drop("total_bedrooms", axis=1)
# 使用中位数填充
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median)


# %%
# 使用sklearn处理缺失值
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_


# %%
housing_num.median().values


# %%
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# %%
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)


# %%
housing_tr.loc[sample_incomplete_rows.index.values]


# %%
# 处理文本和分类属性
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)


# %%
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# %%
housing_cat_1hot.toarray()


# %%
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# %%
cat_encoder.categories_


# %%
housing.columns


# %%
# 自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin

# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# %%
# 另外一种自定义转换器
from sklearn.preprocessing import FunctionTransformer

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)


# %%
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# %%



