from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import plot_confusion_matrix 

data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

LogReg=LogisticRegression(max_iter=1000)
LogReg.fit(X_train_scaled,y_train)
y_pred=LogReg.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred,title='Logistic_Regression_threshold(default)')

y_scores = LogReg.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend()
plt.title("Precision-Recall Tradeoff")
plt.show()

diff=np.abs(precisions[0:-1]-recalls[0:-1])
new_threshold=thresholds[np.argmin(diff)]
print(f"Optimal_threshold:{new_threshold}")

y_pred_new=(y_scores>=new_threshold).astype(int)
plot_confusion_matrix(y_test, y_pred_new,title='Logistic_Regression_threshold(0.05)')

# Save the trained model
model_path_bst = 'models/Logistic_reg_22_22.pkl'
model_path='models/Logistic_reg_32_8.pkl'
joblib.dump(LogReg, model_path)

print(f"✅ Model successfully exported !!!")
# The TP now increased from 49 to 64 and FN reduced to 23 from 38
#  but now the FP has increased to 23 from 10 which is ok but we are still 
# missing 23 of the fraud transactions and hence Logistic regression though 
# having a accuracy of 99.91 is not a optimal model to use for the banks....