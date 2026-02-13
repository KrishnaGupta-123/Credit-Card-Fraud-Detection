import joblib
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

data=joblib.load('data/processed/final_scaled_data.pkl')
data1=joblib.load('data/processed/fraud_data_split.pkl')
X_train=data1['X_train']
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

tree_clf=DecisionTreeClassifier(criterion='entropy',max_depth=20,
                                class_weight='balanced',random_state=1)
tree_clf.fit(X_train_scaled,y_train)
y_pred=tree_clf.predict(X_test_scaled)

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision tree')
plt.show()

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
# plt.figure(figsize=(20, 10))
# plot_tree(
#     tree_clf, 
#     feature_names=X_train.columns, 
#     class_names=['Normal', 'Fraud'], 
#     filled=True, 
#     rounded=True, 
#     fontsize=10
# )
# plt.title("Decision Tree Visualization ")
# plt.show()

# param_grid = {
#     'max_depth': [3, 5, 7, 10,15,20],
#     'criterion': ['gini', 'entropy'],
# }
# grid_search=GridSearchCV(
#     estimator=tree_clf,param_grid=param_grid,
#     scoring='f1',
#     cv=5,
#     n_jobs=-1
# )
# grid_search.fit(X_train_scaled,y_train)
# print(f"Best params:{grid_search.best_params_}")
# print(f"best Score:{grid_search.best_score_}")

features = list(zip(X_train.columns, tree_clf.feature_importances_))
features.sort(key=lambda x: x[1], reverse=True)
print("Feature\t\tImportance")
print("-" * 30)
for column, importance in features:
    print(f"{column}\t{importance:.6f}")