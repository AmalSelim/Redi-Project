#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


# In[52]:


data = pd.read_csv(r"C:\Users\amirh\Desktop\REDI Project\Project REDO Data\Continent_Consumption_TWH.csv", delimiter=',')


# In[53]:


Europe = data['Europe']
Europe

Africa = data['Africa']
Africa

Year = data['Year']
Year


# In[54]:


Africa = data['Africa']
Africa


# In[55]:


Europe = data['Europe']
Europe


# In[56]:


Asia = data['Asia']
Asia


# In[60]:


Pacific = data['Pacific']
Pacific


# In[57]:


BRICS = data ['BRICS']
BRICS


# In[63]:


Latin_America = data ['Latin_America']
Latin_America


# In[66]:


North_America = data ['North_America']
North_America


# In[69]:


OECD = data ['OECD']
OECD


# In[70]:


Middle_East = data ['Middle_East']
Middle_East


# In[71]:


CIS = data ['CIS']
CIS


# In[58]:


World = data ['World']
World


# In[298]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

stacks = ax.stackplot(Year, Europe, Africa, Pacific, Asia, BRICS, Latin_America, North_America, Middle_East, OECD, CIS, World,
                      labels=['Europe', 'Africa', 'Pacific', 'Asia', 'BRICS', 'Latin America', 'North America', 'Middle East', 'OECD', 'CIS', 'World'],
                      alpha=0.5)

ax.set_xlabel('Years')
ax.set_ylabel('Consumption')
ax.set_title('Continent Consumption TWH per each 5 years')


# Add tooltips on the chart
tooltip_positions = [(year, consumption) for year, consumption in zip(Year, total_consumptions) if year % 5 == 0]
tooltip_labels = ['Year: {}\nTotal Consumption: {:.2f}'.format(year, consumption) for year, consumption in tooltip_positions]

for pos, label in zip(tooltip_positions, tooltip_labels):
    ax.annotate(label, xy=pos, xytext=(10, 10), textcoords='offset points', ha='center', fontsize=8, fontweight='bold', color='black')

ax.legend(loc='upper left')

plt.show()


# In[333]:


import pandas as pd
from plotly.graph_objects import Scatter, Figure

data = pd.read_csv("C:/Users/amirh/Desktop/REDI Project/Project REDO Data/Continent_Consumption_TWH.csv", delimiter=',')

# Store the data in a dictionary
data_dict = {
    'Europe': data['Europe'],
    'Africa': data['Africa'],
    'Pacific': data['Pacific'],
    'Asia': data['Asia'],
    'BRICS': data['BRICS'],
    'Latin America': data['Latin_America'],
    'North America': data['North_America'],
    'Middle East': data['Middle_East'],
    'OECD': data['OECD'],
    'CIS': data['CIS'],
    'World': data['World']
}

years = data['Year']

fig = Figure()

for label, values in data_dict.items():
    fig.add_trace(Scatter(
        x=years,
        y=values,
        mode='lines',
        name=label,
        stackgroup='one',
        hovertemplate='%{y:.2f}',
        fill='tonexty',
        line=dict(width=0.5),
    ))

fig.update_layout(
    title='Continent Energy Consumption_TWH',
    xaxis_title='Years',
    yaxis_title='Consumption',
    showlegend=True,
    hovermode='x unified',
)

fig.show()


# In[294]:


data = pd.read_csv(r"C:\Users\amirh\Desktop\REDI Project\Project REDO Data\nonRenewablesTotalPowerGeneration.csv", delimiter=',')
Mode_of_Generation = data['Mode_of_Generation']
Mode_of_Generation


# In[249]:


Contribution_TWh = data['Contribution-TWh']
Contribution_TWh


# In[250]:


cols =['y','b','g','r','k','m', 'c','orange']
textprops = {"fontsize":15}
plt.figure(figsize = (3,2))
explode = [0.1,0.1,0.1,0.3,0.1,0]


# In[251]:


plt.pie(Contribution_TWh, labels= Mode_of_Generation,colors= cols, 
        startangle=90,
        shadow = True,
        autopct='%1.1f%%',
        radius =1.2,
        textprops = textprops,
       rotatelabels = True,
       normalize=True, 
        explode = explode,
        pctdistance = 0.6)
plt.title('Non_Renewables Total Power Generation')
plt.show() 


# In[318]:


import matplotlib.pyplot as plt

countries = ['China', 'United States', 'India', 'Russia', 'Japan', 'Germany', 'Canada', 'South Korea', 'Iran', 'Saudi Arabia']
consumption = [744000000, 391000000, 127000000, 105000000, 105000000, 58000000, 57000000, 54000000, 24000000, 24000000]

fig, ax = plt.subplots()
bars = ax.bar(countries, consumption)

plt.title('Top Ten Electricity Consuming Countries, 2021')
plt.xlabel('Country')
plt.ylabel('Electricity Consumption (kWh)')
plt.xticks(rotation=90)

# Add tooltips to the bar
for bar in bars:
    height = bar.get_height()
    #f format string 
    #,.0f thousands format
    ax.annotate(f'{height:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
              #xytext positions the text 15 points above the arrow
                xytext=(0, 3), 
                # textcoords indicating that the xytext parameter specifies the offset from the arrow.
                textcoords='offset points',
                #horizontal alignment
                ha='center', fontsize=8, rotation='vertical')

plt.show()


# In[332]:


import pandas as pd

df = pd.read_csv(r"C:\Users\amirh\Desktop\REDI Project\Project REDO Data\top20CountriesPowerGeneration.csv", delimiter=',')

# Sort the dataframe by each power generation column by top 10 countries
top_10_hydro = df.sort_values('Hydro_TWh').head(10)
top_10_biofuel = df.sort_values('Biofuel_TWh', ascending=False).head(10)
top_10_solar_pv = df.sort_values('Solar PV_TWh', ascending=False).head(10)
top_10_geothermal = df.sort_values('Geothermal_TWh', ascending=False).head(10)
top_10_total = df.sort_values('Total_TWh', ascending=False).head(10)

# Print the top 10 countries for each power generation category
print("Top 10 Countries for Hydro Power Generation:")
print(top_10_hydro[['Country', 'Hydro_TWh']])
print()

print("Top 10 Countries for Biofuel Power Generation:")
print(top_10_biofuel[['Country', 'Biofuel_TWh']])
print()

print("Top 10 Countries for Solar PV Power Generation:")
print(top_10_solar_pv[['Country', 'Solar PV_TWh']])
print()

print("Top 10 Countries for Geothermal Power Generation:")
print(top_10_geothermal[['Country', 'Geothermal_TWh']])
print()

print("Top 10 Countries for Total Power Generation:")
print(top_10_total[['Country', 'Total_TWh']])


# In[327]:


# Create subplots with 2 rows and 3 columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))

# Bar plot for top 10 countries in Hydro Power Generation
sns.barplot(data=top_10_hydro, x='Country', y='Hydro_TWh', ax=axes[0, 0])
axes[0, 0].set_title('Top 10 Countries for Hydro Power Generation')
axes[0, 0].set_xlabel('Country')
axes[0, 0].set_ylabel('Hydro Power Generation (TWh)')
axes[0, 0].tick_params(axis='x', rotation=90)

# Add tooltips to the bars in the first chart
for bar in axes[0, 0].patches:
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    axes[0, 0].annotate(f'{y:.2f}', (x, y), ha='center', va='bottom')

# Bar plot for top 10 countries in Biofuel Power Generation
sns.barplot(data=top_10_biofuel, x='Country', y='Biofuel_TWh', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Countries for Biofuel Power Generation')
axes[0, 1].set_xlabel('Country')
axes[0, 1].set_ylabel('Biofuel Power Generation (TWh)')
axes[0, 1].tick_params(axis='x', rotation=90)

# Add tooltips to the bars in the second chart
for bar in axes[0, 1].patches:
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    axes[0, 1].annotate(f'{y:.2f}', (x, y), ha='center', va='bottom')

# Bar plot for top 10 countries in Solar PV Power Generation
sns.barplot(data=top_10_solar_pv, x='Country', y='Solar PV_TWh', ax=axes[0, 2])
axes[0, 2].set_title('Top 10 Countries for Solar PV Power Generation')
axes[0, 2].set_xlabel('Country')
axes[0, 2].set_ylabel('Solar PV Power Generation (TWh)')
axes[0, 2].tick_params(axis='x', rotation=90)

# Add tooltips to the bars in the third chart
for bar in axes[0, 2].patches:
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    axes[0, 2].annotate(f'{y:.2f}', (x, y), ha='center', va='bottom')

# Bar plot for top 10 countries in Geothermal Power Generation
sns.barplot(data=top_10_geothermal, x='Country', y='Geothermal_TWh', ax=axes[1, 0])
axes[1, 0].set_title('Top 10 Countries for Geothermal Power Generation')
axes[1, 0].set_xlabel('Country')
axes[1, 0].set_ylabel('Geothermal Power Generation (TWh)')
axes[1, 0].tick_params(axis='x', rotation=90)

# Add tooltips to the bars in the fourth chart
for bar in axes[1, 0].patches:
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    axes[1, 0].annotate(f'{y:.2f}', (x, y), ha='center', va='bottom')

# Bar plot for top 10 countries in Total Power Generation
sns.barplot(data=top_10_total, x='Country', y='Total_TWh', ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Countries for Total Power Generation')
axes[1, 1].set_xlabel('Country')
axes[1, 1].set_ylabel('Total Power Generation (TWh)')
axes[1, 1].tick_params(axis='x', rotation=90)

# Add tooltips to the bars in the fifth chart
for bar in axes[1, 1].patches:
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    axes[1, 1].annotate(f'{y:.2f}', (x, y), ha='center', va='bottom')

# Remove the empty subplot
fig.delaxes(axes[1, 2])

# Adjust the layout
fig.tight_layout()

# Show the plot
plt.show()


# In[331]:


import matplotlib.pyplot as plt

class EnergyConsumption:
    def __init__(self, years, consumption):
        self.years = years
        self.consumption = consumption

    def line_chart(self):
        plt.plot(self.years, self.consumption)
        self.add_value_tooltips()
        plt.xlabel('Years')
        plt.ylabel('Energy Consumption')
        plt.title('Energy Consumption Over Time')
        plt.show()

    def bar_chart(self):
        plt.bar(self.years, self.consumption)
        self.add_value_tooltips()
        plt.xlabel('Years')
        plt.ylabel('Energy Consumption')
        plt.title('Energy Consumption Over Time')
        plt.show()

    def add_value_tooltips(self):
        for x, y in zip(self.years, self.consumption):
            plt.text(x, y, str(y), ha='center', va='bottom')

# Sample data
years = [2015, 2016, 2017, 2018, 2019]
consumption = [1050, 1200, 980, 1130, 895]

# Create an instance of the EnergyConsumption class
energy = EnergyConsumption(years, consumption)

# Generate a line chart
energy.line_chart()

# Generate a bar chart
energy.bar_chart()


# In[324]:


import pandas as pd
import requests
import matplotlib.pyplot as plt

# URL of online data source
url = 'https://media.githubusercontent.com/media/datablist/sample-csv-files/main/files/organizations/organizations-100.csv'

# GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Read the data into a Pandas DataFrame
    df = pd.read_csv(url)

    # Get the top 10 industries by value counts
    top_10_industries = df['Industry'].value_counts().head(10).index

    # Filter the DataFrame to include only the top 10 industries
    filtered_df = df[df['Industry'].isin(top_10_industries)]

    # Create Matplotlib visuals
    fig, ax = plt.subplots()

    # Plot the founded year against the industry for the filtered DataFrame
    for i, industry in enumerate(top_10_industries):
        industry_data = filtered_df[filtered_df['Industry'] == industry]
        ax.plot(industry_data['Founded'], [i] * len(industry_data), marker='o', linestyle='', label=industry, markersize=5)
        for x, y in zip(industry_data['Founded'], [i] * len(industry_data)):
            ax.annotate(f'{x}', (x, y), xytext=(5, -5), textcoords='offset points', fontsize='small')

    ax.set_xlabel('Founded Year')
    ax.set_ylabel('Industry')
    ax.set_title('top_10_industries over Years')
    ax.set_yticks(range(len(top_10_industries)))
    ax.set_yticklabels(top_10_industries)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plot
    plt.show()
else:
    print('Failed to fetch data from the online source.')


# In[330]:


import matplotlib.pyplot as plt

class EnergyConsumptionChart:
    def __init__(self, renewable_data, non_renewable_data, years):
        self.renewable_data = renewable_data
        self.non_renewable_data = non_renewable_data
        self.years = years

    def plot_chart(self):
        fig, ax = plt.subplots()

        # Create the bar chart
        renewable_bars = ax.bar(self.years, self.renewable_data, label='Renewable')
        non_renewable_bars = ax.bar(self.years, self.non_renewable_data, bottom=self.renewable_data, label='Non-renewable')

        # Add labels and title
        plt.xlabel('Year')
        plt.ylabel('Energy Consumption (in units)')
        plt.title('Energy Consumption: Renewable vs. Non-renewable')

        # Add individual values as tooltips
        plt.bar_label(renewable_bars, labels=self.renewable_data, label_type='edge', fontsize=8)
        plt.bar_label(non_renewable_bars, labels=self.non_renewable_data, label_type='edge', fontsize=8)

        plt.legend()
        plt.show()

renewable_data = [50, 60, 70, 80, 90]
non_renewable_data = [150, 140, 130, 120, 110]
years = [2015, 2016, 2017, 2018, 2019]

chart = EnergyConsumptionChart(renewable_data, non_renewable_data, years)
chart.plot_chart()


# In[328]:


import matplotlib.pyplot as plt
import squarify
import matplotlib.patches as mpatches

renewable_industries = ['Solar Energy', 'Wind Energy', 'Hydroelectric Power']
non_renewable_industries = ['Oil', 'Coal', 'Natural Gas']

renewable_values = [250, 400, 300]  
non_renewable_values = [600, 700, 800]

labels = renewable_industries + non_renewable_industries
values = renewable_values + non_renewable_values

# Calculate the total value
renewable_total = sum(renewable_values)
non_renewable_total = sum(non_renewable_values)

# Create the treemap
plt.figure(figsize=(12, 8))
squarify.plot(sizes=values, label=labels, color=colors, alpha=0.8)

# Add text to show the total value for each category
plt.text(0, 0, f"Renewable: {renewable_total}", fontsize=12, weight='bold')
plt.text(0, 5, f"Non-Renewable: {non_renewable_total}", fontsize=12, weight='bold')

plt.axis('off')
plt.title('Renewable Energy vs Non-Renewable Energy', fontsize=16)

# Create legend patches
legend_patches = []
for industry, color in zip(labels, colors):
    patch = mpatches.Patch(color=color, label=industry)
    legend_patches.append(patch)

# Add legend
plt.legend(handles=legend_patches, title='Industry', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()


# In[ ]:




