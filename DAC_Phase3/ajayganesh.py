#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[117]:


product_sales_df = pd.read_csv('C:\\Users\\Abinaya\\OneDrive\\Desktop\\statsfinal.csv')


# In[118]:


product_sales_df.shape


# In[119]:


product_sales_df.columns


# In[120]:


product_sales_df.head(10)


# In[121]:


df.isnull().sum()


# In[6]:


product_sales_df.dtypes


# In[7]:


product_sales_df.isna().sum()


# In[8]:


continuous_columns = ['S-P1', 'S-P2', 'S-P3', 'S-P4']
continuous_data_df = product_sales_df[continuous_columns]
continuous_data_df.head(5)


# In[9]:


continuous_data_df.describe()


# In[10]:


p1 = product_sales_df.drop(['Date','Unnamed: 0'], axis=1)
sns.pairplot(p1)


# In[11]:


corr = p1.corr()
sns.heatmap(data=corr,annot=True)



# In[12]:


discrete_columns = ['Q-P1','Q-P2','Q-P3','Q-P4']
for column in discrete_columns:
    print(product_sales_df[column].value_counts())
    print('\n')


# In[13]:


discrete_columns = ['Q-P1','Q-P2','Q-P3','Q-P4']
for column in discrete_columns:
    product_sales_df[column].value_counts().plot.bar()
    plt.show()


# In[14]:


sns.boxplot(x='Q-P3', y='S-P3', data=product_sales_df)
plt.xlabel('Q-P3')
plt.ylabel('S-P3')
plt.title('Distribution of S-P3 by Q-P3')
plt.show()


# In[15]:


df = product_sales_df
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
except pd.errors.ParserError:
    pass

df = df.dropna(subset=['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date', inplace=True)


products = ['Q-P1', 'Q-P2', 'Q-P3', 'Q-P4']  

for product in products:
    product_data = df[product]

    moving_average = product_data.rolling(window=100).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, moving_average, label=f'{product} - Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Unit Sales')
    plt.title(f'Unit Sales Over Time for {product}')
    plt.grid(True)
    plt.legend()
    plt.show()


# In[16]:


df = product_sales_df
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
except pd.errors.ParserError:
    pass

df = df.dropna(subset=['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date', inplace=True)

products = ['Q-P1', 'Q-P2', 'Q-P3', 'Q-P4']
for product in products:
    product_data = df[product]['2020']

    moving_average = product_data.rolling(window=100).mean()

    # Create a clearer plot of the smoothed data
    plt.figure(figsize=(12, 6))
    plt.plot(product_data.index, moving_average, label=f'{product} - Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Smoothed Unit Sales')
    plt.title(f'Smoothed Unit Sales Over Time for {product} (Year 2020)')
    plt.grid(True)
    plt.legend()
    plt.show()


# In[17]:


import numpy as np
import pandas as pd


# In[18]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[19]:


2 in [1,2,3,4,5]



# In[20]:


7 in [1,2,3,4,5]


# In[21]:


'Devang' in ['Heta','Devang','Yash','Riddhi']


# In[22]:


'Modi' in ['Heta','Devang','Yash','Riddhi']



# In[23]:


import numpy as np
import pandas as pd



# In[25]:


pd.read_csv('C:\\Users\\Abinaya\\OneDrive\\Desktop\\statsfinal.csv')



# In[35]:


A = pd.read_csv("C:\\Users\\Abinaya\\OneDrive\\Desktop\\statsfinal.csv",usecols = ['S-P4'])
A



# In[36]:


11921.36 in A 


# In[37]:


4597 in A 


# In[38]:


11921.36 in A


# In[39]:


5 in A.values


# In[46]:


5 in A.index


# In[41]:


A.values 


# In[42]:


8163.85 in A.values


# In[43]:


import numpy as np 
import pandas as pd


# In[44]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[45]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns=50
sns.set(style="darkgrid")


# In[47]:


df=pd.read_csv("C:\\Users\\Abinaya\\OneDrive\\Desktop\\statsfinal.csv")
df.head(5)


# In[48]:


df.shape


# In[49]:


df.columns


# In[50]:


df.info()


# In[51]:


df.isnull().sum()


# In[52]:


df.dtypes


# In[53]:


df.duplicated().sum()



# In[54]:


df.describe().T


# In[55]:


df.describe().T


# In[56]:


from datetime import datetime as dt
df[df["Date"]=="31-9-2010"]


# In[57]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


# In[58]:


df[df['Date'].isnull()]



# In[59]:


df["Date"].fillna(df["Date"].mean(),inplace=True)


# In[60]:


df['Date'].isnull().sum()



# In[61]:


df.dtypes


# In[62]:


df["month"]=df["Date"].dt.month_name()
df["day"]=df["Date"].dt.day_name()
df["dayoftheweek"]=df["Date"].dt.weekday
df["year"]=df["Date"].dt.year
df.sample()


# In[63]:


df.drop(columns=["Unnamed: 0"],inplace=True)
df.sample()


# In[67]:


for i in df.columns:
    print(i,"---------",df[i].unique())


# In[68]:


df.sample()



# In[69]:


q = df[["Q-P1","Q-P2","Q-P3","Q-P4"]].sum()
print(q)
plt.figure(figsize=(8,8))
plt.pie(q,labels=df[["Q-P1","Q-P2","Q-P3","Q-P4"]].sum().index,shadow=True,autopct="%0.01f%%",textprops={"fontsize":20},wedgeprops={'width': 0.8},explode=[0,0,0,0.3])
plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.8));


# In[70]:


s=df[["S-P1","S-P2","S-P3","S-P4"]].sum()
print(s)
plt.figure(figsize=(8,8))
plt.pie(s,labels=df[["S-P1","S-P2","S-P3","S-P4"]].sum().index,shadow=True,autopct="%0.01f%%",textprops={"fontsize":20},wedgeprops={'width': 0.8},explode=[0,0,0,0.3])
plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.8))


# In[71]:


print(df["month"].value_counts())
plt.figure(figsize=(10,10))
sns.countplot(x="month",data=df,edgecolor="black")
plt.xticks(rotation=90);


# In[72]:


print(df["day"].value_counts())
plt.figure(figsize=(10,10))
sns.countplot(x="day",data=df,edgecolor="black")
plt.xticks(rotation=90);


# In[73]:


print(df["year"].value_counts())
plt.figure(figsize=(10,10))
sns.countplot(x="year",data=df,edgecolor="black")
plt.xticks(rotation=90);


# In[74]:


sns.relplot(x="month",y="S-P1",data=df,kind="line",height=10,color="red")
plt.xticks(rotation=90);
sns.relplot(x="month",y="S-P2",data=df,kind="line",height=10,color="blue")
plt.xticks(rotation=90);
sns.relplot(x="month",y="S-P3",data=df,kind="line",height=10,color="green")
plt.xticks(rotation=90);
sns.relplot(x="month",y="S-P4",data=df,kind="line",height=10,color="purple")
plt.xticks(rotation=90);


# In[75]:


df.groupby("month")[["S-P1","S-P2","S-P3","S-P4"]].sum()


# In[76]:


plt.figure(figsize=(15,15),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="month",y="S-P1",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,2)
sns.barplot(x="month",y="S-P2",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,3)
sns.barplot(x="month",y="S-P3",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,4)
sns.barplot(x="month",y="S-P4",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90)
plt.subplots_adjust(hspace=0.3);


# In[77]:


df.sample()



# In[78]:


df.groupby ("month")[["Q-P1","Q-P2","Q-P3","Q-P4"]].sum()


# In[79]:


plt.figure(figsize=(15,15),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="month",y="Q-P1",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,2)
sns.barplot(x="month",y="Q-P2",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,3)
sns.barplot(x="month",y="Q-P3",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,4)
sns.barplot(x="month",y="Q-P4",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90)
plt.subplots_adjust(hspace=0.3);


# In[80]:


week_t=df[df["dayoftheweek"]<5]
weekend_t=df[df["dayoftheweek"]>=5]
print(week_t.groupby("day")[["S-P1","S-P2","S-P3","S-P4"]].sum())


# In[81]:


plt.figure(figsize=(10,10),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="day",y="S-P1",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,2)
sns.barplot(x="day",y="S-P2",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,3)
sns.barplot(x="day",y="S-P3",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,4)
sns.barplot(x="day",y="S-P4",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45)
plt.subplots_adjust(hspace=0.5);


# In[82]:


print(weekend_t.groupby("day")[["S-P1","S-P2","S-P3","S-P4"]].sum())


# In[83]:


plt.figure(figsize=(10,10),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="day",y="S-P1",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,2)
sns.barplot(x="day",y="S-P2",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,3)
sns.barplot(x="day",y="S-P3",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,4)
sns.barplot(x="day",y="S-P4",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45)
plt.subplots_adjust(hspace=0.5);



# In[84]:


df.groupby("year")[["S-P1","S-P2","S-P3","S-P4"]].agg(["sum"])


# In[85]:


plt.figure(figsize=(10,10),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="year",y="S-P1",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,2)
sns.barplot(x="year",y="S-P2",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,3)
sns.barplot(x="year",y="S-P3",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90);
plt.subplot(2,2,4)
sns.barplot(x="year",y="S-P4",data=df,edgecolor="black",estimator=sum)
plt.xticks(rotation=90)
plt.subplots_adjust(hspace=0.5);



# In[86]:


df[["S-P1","S-P2","S-P3","S-P4"]].agg(["sum","max","min","mean"])


# In[87]:


plt.figure(figsize=(10,10),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="day",y="Q-P1",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,2)
sns.barplot(x="day",y="Q-P2",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,3)
sns.barplot(x="day",y="Q-P3",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,4)
sns.barplot(x="day",y="Q-P4",data=week_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45)
plt.subplots_adjust(hspace=0.5);



# In[88]:


plt.figure(figsize=(10,10),dpi=100)
plt.subplot(2,2,1)
sns.barplot(x="day",y="Q-P1",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,2)
sns.barplot(x="day",y="Q-P2",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,3)
sns.barplot(x="day",y="Q-P3",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45);
plt.subplot(2,2,4)
sns.barplot(x="day",y="Q-P4",data=weekend_t,edgecolor="black",estimator=sum)
plt.xticks(rotation=45)
plt.subplots_adjust(hspace=0.5);


# In[89]:


from wordcloud import WordCloud as word
d=df[["S-P1","S-P2","S-P3","S-P4"]].sum()
wc = word(background_color='white', width=1000, height=600)
wc.generate_from_frequencies(d)
plt.figure(figsize=(15,15),dpi=100)
plt.imshow(wc)
plt.axis('off')
plt.show()



# In[90]:


q=df[["Q-P1","Q-P2","Q-P3","Q-P4"]].sum()
wc = word(background_color='white', width=1000, height=600)
wc.generate_from_frequencies(q)
plt.figure(figsize=(15,15),dpi=100)
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[91]:


import pandas as pd  # library used for data manipulation and analysis
import numpy as np  # library used for working with arrays
import matplotlib.pyplot as plt  # library for plots and visualizations
import seaborn as sns  # library for visualizations

get_ipython().run_line_magic('matplotlib', 'inline')

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[92]:


data = pd.read_csv('C:\\Users\\Abinaya\\OneDrive\\Desktop\\statsfinal.csv')


# In[93]:


data.head(-1)


# In[94]:


data = data.drop(columns=['Unnamed: 0'])


# In[95]:


data.info()


# In[96]:


data.isnull().sum()



# In[97]:


data['Day'] = data['Date'].apply(lambda x: x.split('-')[0])
data['Month'] = data['Date'].apply(lambda x: x.split('-')[1])
data['Year'] = data['Date'].apply(lambda x: x.split('-')[2])
data


# In[98]:


data_reduced = data.query("Year != '2010' and Year != '2023'")


# In[99]:


def plot_bar_chart(df, columns, stri, str1, val):
    # Aggregate sales for each product by year, by sum or mean
    if val == 'sum':
        sales_by_year = df.groupby('Year')[columns].sum().reset_index()
    elif val == 'mean':
        sales_by_year = df.groupby('Year')[columns].mean().reset_index()

    # Melt the data to make it easier to plot
    sales_by_year_melted = pd.melt(sales_by_year, id_vars='Year', value_vars=columns, var_name='Product', value_name='Sales')

    # Create a bar chart
    plt.figure(figsize=(20,4))
    sns.barplot(data=sales_by_year_melted, x='Year', y='Sales', hue='Product') #,palette="cividis")
    plt.xlabel('Year')
    plt.ylabel(stri)
    plt.title(f'{stri} by {str1}')
    plt.xticks(rotation=45)
    plt.show()


# In[100]:


plot_bar_chart(data_reduced, ['Q-P1', 'Q-P2', 'Q-P3', 'Q-P4'],'Total Unit Sales', 'Year', 'sum')

plot_bar_chart(data_reduced, ['Q-P1', 'Q-P2', 'Q-P3', 'Q-P4'],'Mean Unit Sales', 'Year', 'mean')


# In[101]:


plot_bar_chart(data_reduced, ['S-P1', 'S-P2', 'S-P3', 'S-P4'], 'Total Revenue', 'Year', 'sum')

plot_bar_chart(data_reduced, ['S-P1', 'S-P2', 'S-P3', 'S-P4'], 'Mean Revenue', 'Year', 'mean')


# In[102]:


data


# In[103]:


def month_plot():
    fig, ax = plt.subplots()

    # Plot the sales data for each product by month
    data_reduced.groupby('Month')[['Q-P1', 'Q-P2', 'Q-P3', 'Q-P4']].sum().plot(ax=ax)

    # Set the x-axis limits to only show up to December
    ax.set_xlim(left=0, right=13)

    # Set the axis labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Total unit sales')
    ax.set_title('Trend in sales of all four products by month')

    # Show the plot
    plt.show()

month_plot()


# In[104]:


data_reduced['Month'] = data['Month'].replace('9', '09')


# In[105]:


month_plot()



# In[106]:


def month_31_data(df, months):
    m31_data = df[df['Month'].isin(months) & (df['Day'] == '31')]
    return m31_data

_31_months = month_31_data(data_reduced, ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
_31_months


# In[107]:


plot_bar_chart(_31_months, ['Q-P1', 'Q-P2', 'Q-P3', 'Q-P4'], 'Average Units', 'each Month, for 31st', 'mean')


# In[113]:


plot_bar_chart(_31_months, ['S-P1', 'S-P2', 'S-P3', 'S-P4'], 'Average Revenue', 'each Month, for 31st', 'mean')


# In[ ]:




