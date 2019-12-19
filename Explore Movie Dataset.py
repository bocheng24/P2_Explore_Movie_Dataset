#!/usr/bin/env python
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[2]:


df_org = pd.read_csv('data/tmdb-movies.csv')


# In[3]:


df_org.head()


# In[4]:


df_org.tail()


# In[5]:


df_org.sample()


# In[6]:


df_org.dtypes


# In[7]:


df_org.isnull().any()


# In[8]:


df_org.describe()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[9]:


df_org.info()


# In[10]:


df_org.isnull().sum()


# In[11]:


df = df_org.drop(['homepage', 'tagline', 'keywords', 'production_companies', 'imdb_id', 'cast', 'overview'], axis = 1)


# In[12]:


df.isnull().sum()


# In[13]:


df.dropna(inplace = True)


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[14]:


df[['id', 'popularity', 'budget', 'runtime', 'vote_average']]


# In[15]:


df.iloc[np.append(np.arange(0, 20), np.arange(47, 49))]


# In[16]:


df['popularity'].iloc[49 : 60]


# ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[17]:


df.query('popularity > 5')


# In[18]:


df.query('popularity > 5 & release_year > 1996')


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[19]:


df.groupby('release_year').revenue.agg('mean')


# In[20]:


df.groupby('director').popularity.agg('mean').sort_values(ascending = False)


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# In[21]:


pop_20 = df.sort_values('popularity', ascending = False)[['original_title', 'popularity']][:20]
x_ticks = np.arange(0, pop_20['popularity'].max() + 6, 3)

plt.subplots(figsize = (10, 8))
plt.barh(pop_20['original_title'], pop_20['popularity']);

plt.xticks(x_ticks);
plt.title('Top 20 Most Popular Movies');
plt.xlabel('Popularity Score');
plt.ylabel('Movies');


# <b>简要分析</b>：我使用横向的条形图，展示 20 个最受欢迎的电影。前三名的评分比较高，后 9 名的评分都比较接近。

# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[22]:


df['profits'] = df['revenue_adj'] - df['budget_adj']
y_mean = df.groupby('release_year')['profits'].mean()
plt.errorbar(x = y_mean.index, y = y_mean);

plt.title('Movie Profit');
plt.xlabel('Decade');
plt.ylabel('Profit');


# <b>简要分析</b>：我使用折线图展示 从 1960 年代到 2010 年代的利润变化。图中可见在 1960 - 1970 年代初期涨跌幅度较大，利润的最高点出现在 1970 年代中后期，之后利润有大幅下降，并有涨幅，涨跌幅度也比较小，但总体呈现下降趋势。

# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[23]:


top_10_director = df.groupby('director')['director'].count().sort_values(ascending = False)[: 10]
top_10_director


# In[24]:


df_top3 = pd.DataFrame(columns = ['director', 'original_title', 'popularity', 'revenue'])


# In[25]:


for director in top_10_director.index:
    tmp_df = df[df['director'] == director].sort_values('popularity', ascending = False)
    tmp_df = tmp_df[['director', 'original_title', 'popularity', 'revenue']].iloc[: 3]
    tmp_df['revenue'] = np.round(tmp_df['revenue'] / 1000000, 2)
    df_top3 = df_top3.append(tmp_df, ignore_index = True)


# In[26]:


x_rev = np.arange(0, 1100, 100)
fig, ax = plt.subplots(figsize = (10, 10))
ax = sns.barplot(data = df_top3, x = 'revenue', y = 'original_title', hue = 'director');
for bar in ax.patches:
    bar.set_height(0.6)

plt.title('Revenue of Best 3 Movies From Top 10 Most Productive Directors')
plt.xticks(x_rev)
plt.xlabel('Revenue (in Million)')
ax.legend(loc = "upper right", ncol = 2);


# <b>简要分析</b>：我使用横向的条形图，并以导演名字区分长条的颜色，图中票房数字是百万级别。可见，多产的导演票房收益不尽相同，有些多产的导演并且拍的是好莱坞大片，如 Peter Jackson、Steven Spielberg，则他们的票房都会偏高，其中 Peter Jackson 导演的指环王、霍比特人的票房最高。另一方面，一些多产的导演拍的是一些小众电影，如 John Carpenter，则票房偏低

# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[27]:


df_year = df_org.copy()
df_year['release_date'] = pd.to_datetime(df_year['release_date'])
df_year['release_month'] = df_year['release_date'].apply(lambda dt : dt.month)


# In[28]:


df_year = df_year[((df_year['release_year'] > 1967) & (df_year['release_year'] < 2015)) 
                  | ((df_year['release_year'] == 2015) & (df_year['release_month'] < 7))]


# In[29]:


df_year_sub = df_year.sort_values(['release_year', 'release_month'])[['original_title', 'release_year', 'release_month']]


# In[30]:


df_year_size = df_year_sub.groupby(['release_year', 'release_month']).size()
df_year_size = df_year_size.reset_index(name = 'count')


# In[31]:


df_year_size.info()


# In[32]:


df_year_size.head()


# In[33]:


df_year_size.fillna(0, inplace = True)


# In[34]:


plt.subplots(figsize = (12, 12))
sns.barplot(data = df_year_size, x = 'release_month', y = 'count', hue = 'release_year');
y_ticks = np.arange(0, df_year_size['count'].max() + 10, 10)

plt.xlabel('Release Month')
plt.yticks(y_ticks);
plt.ylabel('Number of Movies')
plt.legend(loc = "upper left", bbox_to_anchor = (1, 1), ncol = 5);


# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[35]:


df_genres = df_year.sort_values(['release_year', 'release_month'])[['original_title', 'release_year', 'release_month', 'genres']].dropna()


# In[36]:


df_genres.head()


# In[37]:


df_genres = df_genres[df_genres['genres'].str.contains('Comedy') | df_genres['genres'].str.contains('Drama')]


# In[38]:


def clean_genre(g):
    
    genre = ''
    
    if 'Drama' in g:
        genre = 'Drama'
    if 'Comedy' in g:
        genre = 'Comedy'
    if 'Drama' in g and 'Comedy' in g:
        genre = 'Drama and Comedy'
    
    return genre


# In[39]:


df_genres['genres_x'] = df_genres['genres'].apply(lambda g : clean_genre(g))


# In[40]:


df_genres.head()


# In[41]:


df_genres_size = df_genres.groupby(['release_year', 'release_month', 'genres_x']).size()


# In[42]:


df_genres_size = df_genres_size.reset_index(name = 'count')


# In[43]:


g = sns.FacetGrid(data = df_genres_size, col = 'genres_x')

g.map(sns.barplot, 'release_month', 'count', 'release_year')

plt.legend(loc = "upper left", bbox_to_anchor = (1, 1), ncol = 5);


# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
