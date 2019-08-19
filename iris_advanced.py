
# coding: utf-8

# #### An Iris Extended Case Study Notebook with Machine Learning Model Implementation Using SKLEARN
# 
# ### Notebook by [Studdent Name](#)
# #### Guided by [Daljit Singh](https://www.facebook.com/daljitsinghtrainer)
# #### [CollegeName](Link)

# ### Table-of-contents
# 
# 1. [Introduction](#Introduction)
# 
# 2. [Required libraries](#Required-libraries)
# 
# 3. [The problem domain](#The-problem-domain)
# 
# 4. [Step 1: Answering the question](#Step-1:-Answering-the-question)
# 
# 5. [Step 2: Checking the data](#Step-2:-Checking-the-data)
# 
# 6. [Step 3: Tidying the data](#Step-3:-Tidying-the-data)
# 
# 7. [Step 4: Exploratory analysis](#Step-4:-Exploratory-analysis)
# 
# 8. [Step 5: Classification](#Step-5:-Classification)
# 
# 9. [Step 6: Train Test Splitting](#Training-Test-Spliting)
# 
# 10. [Step 7: Model Creation](#Model-Creation)
# 
# 11. [Step 8: Conclusion](#Conclusion)
# 
# 12. [Step 8: User Input](#User-Input)
# 
# 

# ### Introduction
# 
# [go back to the top ](#Table-of-contents)
# 
# In the time it took you to read this sentence, terabytes of data have been collectively generated across the world — more data than any of us could ever hope to process, much less make sense of, on the machines we're using to read this notebook.
# 
# **In response to this massive influx of data, the field of Data Science has come to the forefront in the past decade. Cobbled together by people from a diverse array of fields — statistics, physics, computer science, design, and many more — the field of Data Science represents our collective desire to understand and harness the abundance of data around us to build a better world.**
# 
# In this notebook, I'm going to go over a basic Python data analysis pipeline from start to finish to show you what a typical data science workflow looks like.
# 

# #### Required libraries
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# If you don't have Python on your computer, you can use the [Anaconda Python distribution](http://continuum.io/downloads) to install most of the Python packages you need. Anaconda provides a simple double-click installer for your convenience.
# 
# This notebook uses several Python packages that come standard with the Anaconda Python distribution. The primary libraries that we'll be using are:
# 
# * **NumPy**: Provides a fast numerical array structure and helper functions.
# * **pandas**: Provides a DataFrame structure to store data in memory and work with it easily and efficiently.
# * **scikit-learn**: The essential Machine Learning package in Python.
# * **matplotlib**: Basic plotting library in Python; most other Python plotting libraries are built on top of it.
# * **Seaborn**: Advanced statistical plotting library.
# 
# To make sure you have all of the packages you need, install them with `conda`:
# 
#     conda install numpy pandas scikit-learn matplotlib seaborn
# 
# `conda` may ask you to update some of them if you don't have the most recent version. Allow it to do so.
# 

# ##The problem domain
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# For the purposes of this exercise, let's pretend we're working for a startup that just got funded to create a smartphone app that automatically identifies species of flowers from pictures taken on the smartphone. We're working with a moderately-sized team of data scientists and will be building part of the data analysis pipeline for this app.
# 
# We've been tasked by our company's Head of Data Science to create a demo machine learning model that takes four measurements from the flowers (sepal length, sepal width, petal length, and petal width) and identifies the species based on those measurements alone.
# 
# 
# <div style="float:left;width:200px;">
# <img src="images/iris_setosa.jpg" width="150px" height="200px"  />
#     <b>Iris Setosa</b>
# </div>
# 
# <div style="float:left;width:200px;">
# <img src="images/irsi_versicolor.jpg" width="150px" height="100px" />
#     <b>Iris Versicolor</b>
# </div>
# <div style="width:200px;">
# <img src="images/iris_virginica.jpg" width="150px" height="200px"  />
#     <b>Iris Virginica</b>
#     </div>
# <br/>
# The four measurements we're using currently come from hand-measurements by the field researchers, but they will be automatically measured by an image processing model in the future.
# 

# ##Step 1: Answering the question
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# The first step to any data analysis project is to define the question or problem we're looking to solve, and to define a measure (or set of measures) for our success at solving that task. The data analysis checklist has us answer a handful of questions to accomplish that, so let's work through those questions.
# 
# Let's do that now. Since we're performing classification, we can use [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) — the fraction of correctly classified flowers — to quantify how well our model is performing. Our company's Head of Data has told us that we should achieve at least 90% accuracy.
# 
# **Thinking about and documenting the problem we're working on is an important step to performing effective data analysis that often goes overlooked.** 
# #### Don't skip it.

# ##Step 2: Checking the data
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# The next step is to look at the data we're working with. Even curated data sets from the government can have errors in them, and it's vital that we spot these errors before investing too much time in our analysis.
# 
# Generally, we're looking to answer the following questions:
# 
# * Is there anything wrong with the data?
# * Are there any quirks with the data?
# * Do I need to fix or remove any of the data?
# 
# Let's start by reading the data into a pandas DataFrame.

# In[18]:


import pandas as pd


# We're in luck! The data seems to be in a usable format.
# 
# The first row in the data file defines the column headers, and the headers are descriptive enough for us to understand what each column represents. The headers even give us the units that the measurements were recorded in, just in case we needed to know at a later point in the project.
# 
# Each row following the first row represents an entry for a flower: four measurements and one class, which tells us the species of the flower.
# 
# **One of the first things we should look for is missing data.** Thankfully, the field researchers already told us that they put a 'NA' into the spreadsheet when they were missing a measurement.
# 
# We can tell pandas to automatically identify missing values if it knows our missing value marker.

# In[19]:



iris_data = pd.read_csv('iris-data.csv')


# In[20]:


iris_data.head()
iris_data['class'].unique()


# Pandas knows to treat rows with 'NA' as missing values.

# Next, it's always a good idea to look at the distribution of our data — especially the outliers.
# 
# Let's start by printing out some summary statistics about the data set.

# In[21]:


iris_data.describe()


# In[22]:


iris_data['class'].unique()


# We can see several useful values from this table. For example, we see that five `petal_width_cm` entries are missing.
# 
# If you ask me, though, tables like this are rarely useful unless we know that our data should fall in a particular range. It's usually better to visualize the data in some way. Visualization makes outliers and errors immediately stand out, whereas they might go unnoticed in a large table of numbers.
# 
# Since we know we're going to be plotting in this section, let's set up the notebook so we can plot inside of it.

# In[23]:


# This line tells the notebook to show plots inside of the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sb


# Next, let's create a **scatterplot matrix**. Scatterplot matrices plot the distribution of each column along the diagonal, and then plot a scatterplot matrix for the combination of each variable. They make for an efficient tool to look for errors in our data.
# 
# We can even have the plotting package color each entry by its class to look for trends within the classes.

# In[24]:


# We have to temporarily drop the rows with 'NA' values
# because the Seaborn plotting function does not know
# what to do with them
sb.pairplot(iris_data.dropna(), hue='class') #Class is column Name


# From the scatterplot matrix, we can already see some issues with the data set:
# 
# 1. There are five classes when there should only be three, meaning there were some coding errors.
# 
# 2. There are some clear outliers in the measurements that may be erroneous: one `sepal_width_cm` entry for `Iris-setosa` falls well outside its normal range, and several `sepal_length_cm` entries for `Iris-versicolor` are near-zero for some reason.
# 
# 3. We had to drop those rows with missing values.
# 
# In all of these cases, we need to figure out what to do with the erroneous data. Which takes us to the next step...

# ##Step 3: Tidying the data
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# Now that we've identified several errors in the data set, we need to fix them before we proceed with the analysis.
# 
# Let's walk through the issues one-by-one.
# 
# >There are five classes when there should only be three, meaning there were some coding errors.
# 
# After talking with the field researchers, it sounds like one of them forgot to add `Iris-` before their `Iris-versicolor` entries. The other extraneous class, `Iris-setossa`, was simply a typo that they forgot to fix.
# 
# Let's use the DataFrame to fix these errors.

# In[25]:


iris_data['class'].unique()


# In[26]:


iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] ='Iris-setosa'

iris_data['class'].unique()


# Much better! Now we only have three class types. Imagine how embarrassing it would've been to create a model that used the wrong classes.
# 
# >There are some clear outliers in the measurements that may be erroneous: one `sepal_width_cm` entry for `Iris-setosa` falls well outside its normal range, and several `sepal_length_cm` entries for `Iris-versicolor` are near-zero for some reason.
# 
# Fixing outliers can be tricky business. It's rarely clear whether the outlier was caused by measurement error, recording the data in improper units, or if the outlier is a real anomaly. For that reason, we should be judicious when working with outliers: if we decide to exclude any data, we need to make sure to document what data we excluded and provide solid reasoning for excluding that data. (i.e., "This data didn't fit my hypothesis" will not stand peer review.)
# 
# In the case of the one anomalous entry for `Iris-setosa`, let's say our field researchers know that it's impossible for `Iris-setosa` to have a sepal width below 2.5 cm. Clearly this entry was made in error, and we're better off just scrapping the entry than spending hours finding out what happened.

# ** Now we have much clear scatter plot ** 

# In[27]:


sb.pairplot(iris_data.dropna(), hue='class')


# In[28]:


# This line drops any 'Iris-setosa' rows with a separal width less than 2.5 cm
iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') |
                          (iris_data['sepal_width_cm'] >= 2.5)]

iris_data.loc[iris_data['class'] == 'Iris-setosa', 
              'sepal_width_cm'].hist()


# Excellent! Now all of our `Iris-setosa` rows have a sepal width greater than 2.5.
# 
# The next data issue to address is the several near-zero sepal lengths for the `Iris-versicolor` rows. Let's take a look at those rows.

# In[29]:


iris_data.loc[ (iris_data['class'] == 'Iris-versicolor') &
              (iris_data['sepal_length_cm'] < 1.0) ]


# How about that? All of these near-zero `sepal_length_cm` entries seem to be off by two orders of magnitude, as if they had been recorded in meters instead of centimeters.
# 
# After some brief correspondence with the field researchers, we find that one of them forgot to convert those measurements to centimeters. Let's do that for them.

# In[30]:


iris_data.loc[(iris_data['class'] == 'Iris-versicolor') &
              (iris_data['sepal_length_cm'] < 1.0),
              'sepal_length_cm'] *= 100.0

iris_data.loc[iris_data['class'] == 'Iris-versicolor', 
              'sepal_length_cm'].hist()


# Phew! Good thing we fixed those outliers. They could've really thrown our analysis off.
# 
# >We had to drop those rows with missing values.
# 
# Let's take a look at the rows with missing values:

# In[31]:


iris_data.loc[(iris_data['sepal_length_cm'].isnull()) |
              (iris_data['sepal_width_cm'].isnull()) |
              (iris_data['petal_length_cm'].isnull()) |
              (iris_data['petal_width_cm'].isnull())]


# It's not ideal that we had to drop those rows, especially considering they're all `Iris-setosa` entries. Since it seems like the missing data is systematic — all of the missing values are in the same column for the same *Iris* type — this error could potentially bias our analysis.
# 
# One way to deal with missing data is **mean imputation**: If we know that the values for a measurement fall in a certain range, we can fill in empty values with the average of that measurement.
# 
# Let's see if we can do that here.

# In[32]:


iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petal_width_cm'].hist()


# Most of the petal widths for `Iris-setosa` fall within the 0.2-0.3 range, so let's fill in these entries with the average measured petal width.

# In[33]:


average_petal_width = iris_data.loc[iris_data['class'] == 
                                    'Iris-setosa', 
                                    'petal_width_cm'].mean()

iris_data.loc[(iris_data['class'] == 'Iris-setosa')&
              (iris_data['petal_width_cm'].isnull()),
                  'petal_width_cm'] = average_petal_width

iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
              (iris_data['petal_width_cm'] == 
               average_petal_width)]


# In[34]:


iris_data.loc[(iris_data['sepal_length_cm'].isnull()) |
              (iris_data['sepal_width_cm'].isnull()) |
              (iris_data['petal_length_cm'].isnull()) |
              (iris_data['petal_width_cm'].isnull())]


# Great! Now we've recovered those rows and no longer have missing data in our data set.
# 
# **Note:** If you don't feel comfortable imputing your data, you can drop all rows with missing data with the `dropna()` call:
# 
#     iris_data.dropna(inplace=True)
# 
# After all this hard work, we don't want to repeat this process every time we work with the data set. Let's save the tidied data file *as a separate file* and work directly with that data file from now on.

# In[1]:


#iris_data.to_csv('iris-data-clean.csv', index=False)
import pandas as pd

iris_data_clean = pd.read_csv('iris-data-clean.csv')


# Now, let's take a look at the scatterplot matrix now that we've tidied the data.

# In[36]:


import seaborn as sb
sb.pairplot(iris_data_clean, hue='class')


# 
# The general takeaways here should be:
# 
# * Make sure your data is encoded properly
# 
# * Make sure your data falls within the expected range, and use domain knowledge whenever possible to define that expected range
# 
# * Deal with missing data in one way or another: replace it if you can or drop it
# 
# * Never tidy your data manually because that is not easily reproducible
# 
# * Use code as a record of how you tidied your data
# 
# * Plot everything you can about the data at this stage of the analysis so you can *visually* confirm everything looks correct

# In[37]:


# We know that our data set should have no missing measurements
assert len(iris_data_clean.loc[(iris_data_clean['sepal_length_cm'].isnull()) |
                               (iris_data_clean['sepal_width_cm'].isnull()) |
                               (iris_data_clean['petal_length_cm'].isnull()) |
                               (iris_data_clean['petal_width_cm'].isnull())]) == 0


# And so on. If any of these expectations are violated, then our analysis immediately stops and we have to return to the tidying stage.

# ##Step 4: Exploratory analysis
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# Now after spending entirely too much time tidying our data, we can start analyzing it!
# 
# Exploratory analysis is the step where we start delving deeper into the data set beyond the outliers and errors. We'll be looking to answer questions such as:
# 
# * How is my data distributed?
# 
# * Are there any correlations in my data?
# 
# * Are there any confounding factors that explain these correlations?
# 
# This is the stage where we plot all the data in as many ways as possible. Create many charts, but don't bother making them pretty — these charts are for internal use.
# 
# Let's return to that scatterplot matrix that we used earlier.

# In[38]:


sb.pairplot(iris_data_clean)


# Our data is normally distributed for the most part, which is great news if we plan on using any modeling methods that assume the data is normally distributed.
# 
# There's something strange going on with the petal measurements. Maybe it's something to do with the different `Iris` types. Let's color code the data by the class again to see if that clears things up.

# In[4]:


import seaborn as sb
sb.pairplot(iris_data_clean, hue='class')


# Sure enough, the strange distribution of the petal measurements exist because of the different species. This is actually great news for our classification task since it means that the petal measurements will make it easy to distinguish between `Iris-setosa` and the other `Iris` types.
# 
# Distinguishing `Iris-versicolor` and `Iris-virginica` will prove more difficult given how much their measurements overlap.
# 
# There are also correlations between petal length and petal width, as well as sepal length and sepal width. The field biologists assure us that this is to be expected: Longer flower petals also tend to be wider, and the same applies for sepals.
# 
# We can also make **violin plots** of the data to compare the measurement distributions of the classes. Violin plots contain the same information as [box plots](https://en.wikipedia.org/wiki/Box_plot), but also scales the box according to the density of the data.

# In[40]:


plt.figure(figsize=(10, 10))

for column_index, column in enumerate(iris_data_clean.columns):
    if column == 'class':
        continue
    plt.subplot(2, 2, column_index + 1)
    sb.violinplot(x='class', y=column, data=iris_data_clean)


# Enough work with the data. Let's get to modeling.

# ##Step 5: Classification
# 
# [[ go back to the top ]](#Table-of-contents)
# 
# Wow, all this work and we *still* haven't modeled the data!
# 
# As tiresome as it can be, tidying and exploring our data is a vital component to any data analysis. If we had jumped straight to the modeling step, we would have created a faulty classification model.
# 
# Remember: **Bad data leads to bad models.** Always check your data first.
# 
# <hr />
# 
# 
# 
# A **training set** is a random subset of the data that we use to train our models.
# 
# A **testing set** is a random subset of the data (mutually exclusive from the training set) that we use to validate our models on unforseen data.
# 
# Especially in sparse data sets like ours, it's easy for models to **overfit** the data: The model will learn the training set so well that it won't be able to handle most of the cases it's never seen before. This is why it's important for us to build the model with the training set, but score it with the testing set.
# 
# Note that once we split the data into a training and testing set, we should treat the testing set like it no longer exists: We cannot use any information from the testing set to build our model or else we're cheating.
# 
# Let's set up our data first.

# In[41]:


iris_data_clean = pd.read_csv('iris-data-clean.csv')



all_inputs = iris_data_clean[['sepal_length_cm', 
                              'sepal_width_cm',
                             'petal_length_cm', 
                              'petal_width_cm']].values

# Similarly, we can extract the classes
all_classes = iris_data_clean['class'].values

# Make sure that you don't mix up the order of the entries
# all_inputs[5] inputs should correspond to the class in all_classes[5]

# Here's what a subset of our inputs looks like:
all_inputs[:5]


# Now our data is ready to be split.

# In[42]:


all_classes[:5]


# # Training-Test-Spliting
# [[ go back to the top ]](#Table-of-contents)

# In[43]:


from sklearn.model_selection import train_test_split

(training_inputs,
 testing_inputs,
 training_classes,
     testing_classes) = train_test_split(all_inputs, 
                                         all_classes, 
                                     train_size=0.75, 
                                     random_state=1)


# # Model-Creation
# [[ go back to the top ]](#Table-of-contents)

# In[44]:


from sklearn.tree import DecisionTreeClassifier

# Create the classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
decision_tree_classifier.fit(training_inputs, 
                             training_classes)
# Validate the classifier on the testing set using classification accuracy
decision_tree_classifier.score
(testing_inputs, testing_classes)

#predict
prediction = decision_tree_classifier.predict(testing_inputs)

#probabilities
probs = decision_tree_classifier.predict_proba(testing_inputs)

#Print Predition
print(prediction[:3])


# In[45]:


print(probs[:5])


# # Conclusion
# [[ go back to the top ]](#Table-of-contents)
# Heck yeah! Our model achieves 97% classification accuracy without much effort.
# 
# However, there's a catch: Depending on how our training and testing set was sampled, our model can achieve anywhere from 80% to 100% accuracy:

# # User-Input
# [[ go back to the top ]](#Table-of-contents)

# In[46]:


#Taking User Imput And Prediction result of Iris-Flowers

import numpy as np
#take input from user
sl=input("sepal length")
pl=input("petal length")
sw=input("sepal width")
pw=input("petal width")

ui=np.array([[sl,sw,pl,pw]])

prediction_result=decision_tree_classifier.predict(ui)

for i in prediction_result:
    print("Your dimension result is = : "+i.upper())
    

