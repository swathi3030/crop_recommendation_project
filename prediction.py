import numpy
import joblib

model=joblib.load('knncropmodel.pkl')

testdata=numpy.array([[60,54,19,18.74826712,62.49878458,6.417820493,70.23401597]])

result=model.predict(testdata)

print(f"Result of prediction={result[0]}")