#!/usr/bin/env python
# coding: utf-8

# #  Data preprocessing

# ## Load data

# In[80]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


# In[2]:


df=pd.read_csv('C://Users//abc//Documents//Downloads//nst-est2019-alldata.csv')


# In[7]:


df_states=df[['NAME','POPESTIMATE2010',	'POPESTIMATE2011',	'POPESTIMATE2012',	'POPESTIMATE2013',	'POPESTIMATE2014',	'POPESTIMATE2015',	'POPESTIMATE2016',	'POPESTIMATE2017',	'POPESTIMATE2018',	'POPESTIMATE2019',	'NPOPCHG_2010',	'NPOPCHG_2011',	'NPOPCHG_2012',	'NPOPCHG_2013',	'NPOPCHG_2014',	'NPOPCHG_2015',	'NPOPCHG_2016',	'NPOPCHG_2017',	'NPOPCHG_2018',	'NPOPCHG_2019',	'BIRTHS2010',	'BIRTHS2011',	'BIRTHS2012',	'BIRTHS2013',	'BIRTHS2014',	'BIRTHS2015',	'BIRTHS2016',	'BIRTHS2017',	'BIRTHS2018',	'BIRTHS2019',	'DEATHS2010',	'DEATHS2011',
              'DEATHS2012',	'DEATHS2013',	'DEATHS2014',	'DEATHS2015',	'DEATHS2016',	'DEATHS2017',	'DEATHS2018',	'DEATHS2019',	'INTERNATIONALMIG2010',	'INTERNATIONALMIG2011',	'INTERNATIONALMIG2012',	'INTERNATIONALMIG2013',	'INTERNATIONALMIG2014',	'INTERNATIONALMIG2015',	'INTERNATIONALMIG2016',	'INTERNATIONALMIG2017',	'INTERNATIONALMIG2018',	'INTERNATIONALMIG2019',	'DOMESTICMIG2010',	'DOMESTICMIG2011',	'DOMESTICMIG2012',	'DOMESTICMIG2013',	'DOMESTICMIG2014',	'DOMESTICMIG2015',	'DOMESTICMIG2016',	'DOMESTICMIG2017',	
              'DOMESTICMIG2018',	'DOMESTICMIG2019']]


# In[8]:


df_regions=df_states[(df['NAME']=='United States')&(df['NAME']=='Northeast Region')
                    &(df['NAME']=='Midwest Region')&(df['NAME']=='South Region')
                    &(df['NAME']=='West Region')]
df_states=df_states[(df['NAME']!='United States')&(df['NAME']!='Northeast Region')
                    &(df['NAME']!='Midwest Region')&(df['NAME']!='South Region')
                    &(df['NAME']!='West Region')]


# In[10]:


df_states.reset_index(drop=True, inplace=True)
df_regions.reset_index(drop=True, inplace=True)


# In[46]:


df_total_pop_states=df_states[['NAME', 'POPESTIMATE2010', 'POPESTIMATE2011', 'POPESTIMATE2012',
       'POPESTIMATE2013', 'POPESTIMATE2014', 'POPESTIMATE2015',
       'POPESTIMATE2016', 'POPESTIMATE2017', 'POPESTIMATE2018',
       'POPESTIMATE2019']]
df_net_pop_change_states=df_states[['NAME','NPOPCHG_2010', 'NPOPCHG_2011', 'NPOPCHG_2012',
       'NPOPCHG_2013', 'NPOPCHG_2014', 'NPOPCHG_2015', 'NPOPCHG_2016',
       'NPOPCHG_2017', 'NPOPCHG_2018', 'NPOPCHG_2019']]
df_total_births_states=df_states[['NAME', 'BIRTHS2010',
       'BIRTHS2011', 'BIRTHS2012', 'BIRTHS2013', 'BIRTHS2014', 'BIRTHS2015',
       'BIRTHS2016', 'BIRTHS2017', 'BIRTHS2018', 'BIRTHS2019']]
df_total_deaths_states=df_states[['NAME','DEATHS2010',
       'DEATHS2011', 'DEATHS2012', 'DEATHS2013', 'DEATHS2014', 'DEATHS2015',
       'DEATHS2016', 'DEATHS2017', 'DEATHS2018', 'DEATHS2019']]
df_int_mig_states=df_states[['NAME','INTERNATIONALMIG2010', 'INTERNATIONALMIG2011', 'INTERNATIONALMIG2012',
       'INTERNATIONALMIG2013', 'INTERNATIONALMIG2014', 'INTERNATIONALMIG2015',
       'INTERNATIONALMIG2016', 'INTERNATIONALMIG2017', 'INTERNATIONALMIG2018',
       'INTERNATIONALMIG2019']]
df_dom_mig_states=df_states[['NAME','DOMESTICMIG2010', 'DOMESTICMIG2011',
       'DOMESTICMIG2012', 'DOMESTICMIG2013', 'DOMESTICMIG2014',
       'DOMESTICMIG2015', 'DOMESTICMIG2016', 'DOMESTICMIG2017',
       'DOMESTICMIG2018', 'DOMESTICMIG2019']]


# In[48]:


new_columns = ['states', '2010', '2011', '2012', '2013', '2014', '2015', '2016','2017', '2018', '2019']
df_total_pop_states.columns= new_columns
df_net_pop_change_states.columns= new_columns
df_total_births_states.columns= new_columns
df_total_deaths_states.columns=new_columns
df_int_mig_states.columns= new_columns
df_dom_mig_states.columns= new_columns


# In[50]:


# Reshape the DataFrame
df_total_pop_states_reshaped = pd.melt(df_total_pop_states, id_vars=['states'], var_name='year', value_name='population')

# Convert 'year' column values to integers
df_total_pop_states_reshaped['states'] = df_total_pop_states_reshaped['states'].astype(str)
df_total_pop_states_reshaped['year'] = df_total_pop_states_reshaped['year'].astype(int)
df_total_pop_states_reshaped['population'] = df_total_pop_states_reshaped['population'].astype(int)

df_total_pop_states=df_total_pop_states_reshaped

# Reshape the DataFrame
df_net_pop_change_states = pd.melt(df_net_pop_change_states, id_vars=['states'], var_name='year', value_name='population_change')

# Convert 'year' column values to integers
df_net_pop_change_states['states'] = df_net_pop_change_states['states'].astype(str)
df_net_pop_change_states['year'] = df_net_pop_change_states['year'].astype(int)
df_net_pop_change_states['population_chnage'] = df_net_pop_change_states['population_change'].astype(int)

# Reshape the DataFrame
df_total_births_states = pd.melt(df_total_births_states, id_vars=['states'], var_name='year', value_name='births')

# Convert 'year' column values to integers
df_total_births_states['states'] = df_total_births_states['states'].astype(str)
df_total_births_states['year'] = df_total_births_states['year'].astype(int)
df_total_births_states['births'] = df_total_births_states['births'].astype(int)

# Reshape the DataFrame
df_total_deaths_states = pd.melt(df_total_deaths_states, id_vars=['states'], var_name='year', value_name='deaths')

# Convert 'year' column values to integers
df_total_deaths_states['states'] = df_total_deaths_states['states'].astype(str)
df_total_deaths_states['year'] = df_total_deaths_states['year'].astype(int)
df_total_deaths_states['deaths'] = df_total_deaths_states['deaths'].astype(int)

# Reshape the DataFrame
df_int_mig_states = pd.melt(df_int_mig_states, id_vars=['states'], var_name='year', value_name='int_mig')

# Convert 'year' column values to integers
df_int_mig_states['states'] = df_int_mig_states['states'].astype(str)
df_int_mig_states['year'] = df_int_mig_states['year'].astype(int)
df_int_mig_states['int_mig'] = df_int_mig_states['int_mig'].astype(int)

# Reshape the DataFrame
df_dom_mig_states = pd.melt(df_dom_mig_states, id_vars=['states'], var_name='year', value_name='dom_mig')

# Convert 'year' column values to integers
df_dom_mig_states['states'] = df_dom_mig_states['states'].astype(str)
df_dom_mig_states['year'] = df_dom_mig_states['year'].astype(int)
df_dom_mig_states['dom_mig'] = df_dom_mig_states['dom_mig'].astype(int)


# In[53]:


print(df_total_pop_states.shape)
print(df_net_pop_change_states.shape)
print(df_total_births_states.shape)
print(df_total_deaths_states.shape)
print(df_int_mig_states.shape)
print(df_dom_mig_states.shape)


# In[54]:


final_df = pd.concat([df_total_pop_states,df_net_pop_change_states,df_total_births_states,df_total_deaths_states,
                     df_int_mig_states,df_dom_mig_states], axis=1)


# In[55]:


final_df.head()


# In[56]:


final_df.columns


# In[57]:


final_df.columns=['states', 'year', 'population', 'states_1', 'year_1', 'population_change',
       'population_chnage', 'states_2', 'year_2', 'births', 'states_3', 'year_3',
       'deaths', 'states_4', 'year_4', 'int_mig', 'states_5', 'year_5', 'dom_mig']


# In[58]:


final_df=final_df[['states','year','population','population_change','births','deaths','int_mig','dom_mig']]


# In[59]:


final_df.head()


# ## Plots

# In[62]:


import altair as alt
import plotly.express as px
import streamlit as st


# In[63]:


#dashboard page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


# In[65]:


#creating sidebar

with st.sidebar:
    st.title('🏂 US Population Dashboard')
    
    year_list = list(final_df.year.unique())[::-1]
    
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
    df_selected_year = final_df[final_df.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


# ### Heatmap

# In[66]:


def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    # height=300
    return heatmap


# ### Choropleth

# In[75]:


def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                               color_continuous_scale=input_color_theme,
                               range_color=(0, max(df_selected_year.population)),
                               scope="usa",
                               labels={'population':'Population'}
                              )
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth


# In[72]:


def calculate_population_difference(input_df, input_year):
    selected_year_data = input_df[input_df['year'] == input_year].reset_index()
    previous_year_data = input_df[input_df['year'] == input_year - 1].reset_index()
    selected_year_data['population_difference'] = selected_year_data.population.sub(previous_year_data.population, fill_value=0)
    return pd.concat([selected_year_data.states, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)


# In[68]:


def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    
    source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
    source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
    return plot_bg + plot + text


# In[69]:


def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'


# In[70]:


#dashboard layout
col = st.columns((1.5, 4.5, 2), gap='medium')


# In[73]:


with col[0]:
    st.markdown('#### Gains/Losses')

    df_reshaped=final_df[['states','year','population']]
    df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_year)

    if selected_year > 2010:
        first_state_name = df_population_difference_sorted.states.iloc[0]
        first_state_population = format_number(df_population_difference_sorted.population.iloc[0])
        first_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[0])
    else:
        first_state_name = '-'
        first_state_population = '-'
        first_state_delta = ''
    st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)

    if selected_year > 2010:
        last_state_name = df_population_difference_sorted.states.iloc[-1]
        last_state_population = format_number(df_population_difference_sorted.population.iloc[-1])   
        last_state_delta = format_number(df_population_difference_sorted.population_difference.iloc[-1])   
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)

    
    st.markdown('#### States Migration')

    if selected_year > 2010:
        # Filter states with population difference > 50000
        # df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference_absolute > 50000]
        df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference > 50000]
        df_less_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference < -50000]
        
        # % of States with population difference > 50000
        states_migration_greater = round((len(df_greater_50000)/df_population_difference_sorted.states.nunique())*100)
        states_migration_less = round((len(df_less_50000)/df_population_difference_sorted.states.nunique())*100)
        donut_chart_greater = make_donut(states_migration_greater, 'Inbound Migration', 'green')
        donut_chart_less = make_donut(states_migration_less, 'Outbound Migration', 'red')
    else:
        states_migration_greater = 0
        states_migration_less = 0
        donut_chart_greater = make_donut(states_migration_greater, 'Inbound Migration', 'green')
        donut_chart_less = make_donut(states_migration_less, 'Outbound Migration', 'red')

    migrations_col = st.columns((0.2, 1, 0.2))
    with migrations_col[1]:
        st.write('Inbound')
        st.altair_chart(donut_chart_greater)
        st.write('Outbound')
        st.altair_chart(donut_chart_less)


# In[78]:


with col[1]:
    st.markdown('#### Total Population')
    
    choropleth = make_choropleth(df_reshaped, 'states', 'population', selected_color_theme)
    st.plotly_chart(choropleth, use_container_width=True)
    
    heatmap = make_heatmap(df_reshaped, 'year', 'states', 'population', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)


# In[79]:


with col[2]:
    st.markdown('#### Top States')

    st.dataframe(df_reshaped,
                 column_order=("states", "population"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "states": st.column_config.TextColumn(
                        "States",
                    ),
                    "population": st.column_config.ProgressColumn(
                        "Population",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_year_sorted.population),
                     )}
                 )
    
    with st.expander('About', expanded=True):
        st.write('''
            - Data: [U.S. Census Bureau](<https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html>).
            - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
            ''')


# In[ ]:




