import joblib
from sklearn.tree import DecisionTreeClassifier,plot_tree
from utils import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

data=joblib.load('data/processed/smote_data.pkl')
X_train_resampled=data['X_train_resampled']
y_train_resampled=data['y_train_resampled']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

tree_clf=DecisionTreeClassifier(max_depth=20,
                                random_state=1)
tree_clf.fit(X_train_resampled,y_train_resampled)
y_pred=tree_clf.predict(X_test_scaled)

plot_confusion_matrix(y_test,y_pred,title='Decision Tree Smote')

param_grid = {
    'max_depth': [10,15,20],
    'min_samples_split':[2,3]
}
grid_search=GridSearchCV(
    estimator=tree_clf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train_resampled,y_train_resampled)
print(f"Best params:{grid_search.best_params_}")
print(f"best Score:{grid_search.best_score_}")


y_pred_grid=grid_search.predict(X_test_scaled)
plot_confusion_matrix(y_test,y_pred_grid,title='Decision Tree Smote GridSearch')

model_path='models/Decision_Trees_Smote_19_253.pkl'
joblib.dump(tree_clf, model_path)
print(f"✅ Model successfully exported !!!")