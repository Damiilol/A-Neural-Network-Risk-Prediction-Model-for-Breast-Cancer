import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tempfile
import sklearn
import tensorflow as tf
from sklearn.calibration import calibration_curve
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import Callback,ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras_visualizer import visualizer 
import keras.backend as K
from keras.utils.vis_utils import plot_model
from ann_visualizer.visualize import ann_viz
from keras_visualizer import visualizer 
from sklearn.metrics import precision_recall_fscore_support


#data loading into a dataframe

data = pd.read_csv('C:/Users/HP/Documents/riskfactor1 (version 1).csv')

    
print('Dimension of the dataset : ', data.shape)

print(data.head())


print(data.head)
data.describe()
data.hist(figsize = (13,13))
plt.show()
#feature importance
X = data.iloc[:,0:13]  
y = data.iloc[:,-12]

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#correlation
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(13,13))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#data = data.rename(columns={"breast_cancer_history":'Label'})
#print(data.dtypes)

no_cancer = len(data[data['breast_cancer_history'] == 1])
print(no_cancer)
non_cancer = len(data[data['breast_cancer_history'] == 0])
print(non_cancer)
un_cancer = len(data[data['breast_cancer_history'] == 9])
print(un_cancer)
total = non_cancer + no_cancer + un_cancer
print('Total: {}\n    Women with Cancer: {} ({:.2f}% of total)\n'.format(
    total, no_cancer, 100 * no_cancer / total))
print('Total: {}\n    Women no with Cancer: {} ({:.2f}% of total)\n'.format(
    total, non_cancer, 100 * non_cancer / total))
print('Total: {}\n    Women unknown: {} ({:.2f}% of total)\n'.format(
    total, un_cancer, 100 * un_cancer / total))
    
 #pre-processing
data.drop(data[data['breast_cancer_history'] >=9].index, inplace = True)

print(data.shape)

data.drop(data[data['age_first_birth'] >=9].index, inplace = True)

print(data.shape)

data.drop(data[data['first_degree_hx'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['age_menarche'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['bmi_group'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['menopaus'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['race_eth'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['biophx'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['BIRADS_breast_density'] >=9].index, inplace = True)
print(data.shape)

data.drop(data[data['current_hrt'] >=9].index, inplace = True)
print(data.shape)

#renaming the target variable
data = data.rename(columns={"breast_cancer_history":'Label'})
print(data.dtypes)

no_cancer = len(data[data['Label'] == 1])
print(no_cancer)
non_cancer = len(data[data['Label'] == 0])
print(non_cancer)
total = non_cancer + no_cancer
print('Total: {}\n    Women with Cancer: {} ({:.2f}% of total)\n'.format(
    total, no_cancer, 100 * no_cancer / total))
print('Total: {}\n    Women no with Cancer: {} ({:.2f}% of total)\n'.format(
    total, non_cancer, 100 * non_cancer / total))
#cleaned_data2= data.copy()
#cleaned_data2.pop('count')

#pre-processing
cleaned_data = data.copy()
cleaned_data.pop('year')

#data splitting
train_X, test_X = train_test_split(cleaned_data, test_size=0.2)
train_X, val_X = train_test_split(train_X, test_size=0.2)

train_labels = np.array(train_X.pop('Label'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_X.pop('Label'))
test_labels = np.array(test_X.pop('Label'))

train_features = np.array(train_X)
val_features = np.array(val_X)
test_features = np.array(test_X)

#feature scaling
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    
#defining the metrics
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      get_f1
]

#model functon definition
def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          256, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
      #model.add(Dense(64, input_dim= (train_features.shape[-1],), activation='relu'))
      #
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model
 
 EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model()
model.summary()

model.predict(train_features[:10])

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

initial_bias = np.log([no_cancer/non_cancer])
initial_bias

model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)
    
    model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)
    
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  plt.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)

#training the model
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels))
    
def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
plot_metrics(baseline_history)

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Women not at risk (True Negatives): ', cm[0][0])
  print('Women not at risk detected (False Positives): ', cm[0][1])
  print('Women at risk not wrongly detected (False Negatives): ', cm[1][0])
  print('Women at risk detected (True Positives): ', cm[1][1])
  print('Total risk cases: ', np.sum(cm[1]))
  
#testing the model
baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)

