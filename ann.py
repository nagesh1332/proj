import pandas as pd


df=pd.read_excel('COVID-19.xlsx')
y = df['Corona result']
y_train_cat = pd.get_dummies(y)
df.columns
X = df.drop(['Sno','gender','age','body temperature'] , axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from keras.models import Sequential
model  =  Sequential()
from keras.layers import Dense
model.add(Dense(units=6 , input_shape=(17,), 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.add(Dense(units=5 , 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.summary()
model.add(Dense(units=4, 
                activation='relu', 
                kernel_initializer='he_normal' ))
model.add(Dense(units=3, activation='softmax'))
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_pred=model.fit(X_train,y_train_cat, epochs=21)
model.save('modelsave.h5')
