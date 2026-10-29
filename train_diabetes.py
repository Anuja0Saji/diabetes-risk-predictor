"""
Diabetes Risk Prediction and Patient Grouping
Run:  python train_diabetes.py        (needs data/diabetes.csv)
"""
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
DATA, OUT = "data/diabetes.csv", "outputs"
os.makedirs(OUT, exist_ok=True)
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"]
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}", dpi=130)
    plt.close()


def make_prep():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])


def pipe(clf):
    return Pipeline([("prep", make_prep()), ("clf", clf)])


# ===== STEP 1: Load =====
if not os.path.exists(DATA):
    raise SystemExit("Put diabetes.csv inside the data folder first.")
df = pd.read_csv(DATA)
print("STEP 1: shape", df.shape)

# ===== STEP 2: Understand =====
print("\nSTEP 2: data understanding")
print(df.describe().T)
print("Duplicates:", df.duplicated().sum())
print("Target balance:\n", df["Outcome"].value_counts())
print("Zero values (impossible for these columns):\n", (df[ZERO_COLS] == 0).sum())

# ===== STEP 3: Preprocessing =====
print("\nSTEP 3: preprocessing")
df = df.drop_duplicates()
df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)      # zeros mean "not measured"
print("Missing values now:\n", df.isnull().sum()[lambda s: s > 0])
# median imputation and scaling are done inside the pipeline (after the split) to avoid data leakage

# ===== STEP 4: EDA =====
sns.countplot(x="Outcome", data=df)
save_fig("eda_target.png")
df[FEATURES].hist(bins=20, figsize=(12, 8))
save_fig("eda_histograms.png")
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.ravel(), FEATURES):
    sns.boxplot(x="Outcome", y=c, data=df, ax=ax)
save_fig("eda_boxplots.png")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
save_fig("eda_correlation.png")

# ===== STEP 5: Split =====
X, y = df[FEATURES], df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("\nSTEP 5: train", X_train.shape, "test", X_test.shape)

# ===== STEP 6: Models =====
models = {"Logistic Regression": pipe(LogisticRegression(max_iter=1000)),
          "Decision Tree": pipe(DecisionTreeClassifier(max_depth=4, random_state=42)),
          "Naive Bayes": pipe(GaussianNB()),
          "KNN": pipe(KNeighborsClassifier(n_neighbors=9)),
          "SVM": pipe(SVC(probability=True, random_state=42)),
          "Random Forest": pipe(RandomForestClassifier(n_estimators=200, random_state=42))}
grid = GridSearchCV(pipe(RandomForestClassifier(random_state=42)),
                    {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 4, 8],
                     "clf__min_samples_leaf": [1, 3]}, cv=5, scoring="f1", n_jobs=-1).fit(X_train, y_train)
print("Best Random Forest settings:", grid.best_params_)
models["Random Forest (tuned)"] = clone(grid.best_estimator_)

# ===== STEP 7: Cross-validation comparison =====
print("\nSTEP 7: 5-fold cross-validation")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []
for name, m in models.items():
    r = cross_validate(m, X_train, y_train, cv=cv, scoring=metrics)
    rows.append({"Model": name, **{k: round(r[f"test_{k}"].mean(), 3) for k in metrics}})
results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
print(results)
results.to_csv(f"{OUT}/model_comparison.csv", index=False)
results.set_index("Model")[metrics].plot(kind="bar", figsize=(11, 5), ylim=(0.3, 1))
plt.xticks(rotation=30, ha="right")
save_fig("model_comparison.png")

# ===== STEP 8: Final test =====
best_name = results.loc[0, "Model"]
best = models[best_name].fit(X_train, y_train)
report = classification_report(y_test, best.predict(X_test), target_names=["No diabetes", "Diabetes"])
print(f"\nSTEP 8: best model = {best_name}\n{report}")
open(f"{OUT}/test_report.txt", "w").write(f"Best model: {best_name}\n\n{report}")
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, display_labels=["No", "Yes"])
save_fig("confusion_matrix.png")
RocCurveDisplay.from_estimator(best, X_test, y_test)
save_fig("roc_curve.png")
joblib.dump(best, f"{OUT}/model.joblib")
open(f"{OUT}/best_model_name.txt", "w").write(best_name)

# ===== STEP 9: Feature importance =====
rf = clone(models["Random Forest"]).fit(X_train, y_train)
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", figsize=(8, 5), title="Feature importance (Random Forest)")
save_fig("feature_importance.png")
print("\nSTEP 9: feature importance\n", imp.sort_values(ascending=False).round(3))

# ===== STEP 10: Clustering =====
print("\nSTEP 10: clustering")
prep_cl = make_prep().fit(X)
Xc = prep_cl.transform(X)
sil = {k: silhouette_score(Xc, KMeans(k, n_init=10, random_state=42).fit_predict(Xc)) for k in range(2, 7)}
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
K = max(range(3, 6), key=lambda k: sil[k])
km = KMeans(K, n_init=10, random_state=42).fit(Xc)


def sil_score(lbl):
    ok = lbl != -1
    return round(silhouette_score(Xc[ok], lbl[ok]), 3) if len(set(lbl[ok])) > 1 else np.nan


comp = pd.DataFrame({"Algorithm": ["K-Means", "Hierarchical", "DBSCAN"],
                     "Silhouette": [sil_score(km.labels_), sil_score(AgglomerativeClustering(K).fit_predict(Xc)),
                                    sil_score(DBSCAN(eps=1.2, min_samples=8).fit_predict(Xc))]})
print(comp)
comp.to_csv(f"{OUT}/clustering_comparison.csv", index=False)
dfc = df.assign(Group=km.labels_)
profile = dfc.groupby("Group").agg(
    Patients=("Age", "size"), Avg_Age=("Age", "mean"), Avg_Glucose=("Glucose", "mean"),
    Avg_BMI=("BMI", "mean"), Avg_BloodPressure=("BloodPressure", "mean"),
    Diabetes_Rate_Percent=("Outcome", lambda s: s.mean() * 100)).round(1)
print(profile)
profile.to_csv(f"{OUT}/cluster_profile.csv")
pcs = PCA(2).fit_transform(Xc)
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=km.labels_, palette="deep")
plt.title("Patient groups (PCA 2D)")
save_fig("clusters_pca.png")
joblib.dump(Pipeline([("prep", prep_cl), ("km", km)]), f"{OUT}/cluster_model.joblib")

# ===== STEP 11: Outliers =====
print("\nSTEP 11: outliers")
iso = IsolationForest(contamination=0.05, random_state=42).fit_predict(Xc) == -1
q1, q3 = df["Insulin"].quantile([0.25, 0.75])
iqr = (df["Insulin"] > q3 + 1.5 * (q3 - q1)).values
print("Isolation Forest:", iso.sum(), "| IQR on Insulin:", iqr.sum(), "| both:", (iso & iqr).sum())
print("Diabetes rate among outliers: %.1f%%" % (df["Outcome"].values[iso].mean() * 100))

# ===== STEP 12: Association rules =====
print("\nSTEP 12: association rules")
try:
    from mlxtend.frequent_patterns import apriori, association_rules

    d = df.copy()
    d["Glucose"] = pd.cut(d["Glucose"], [0, 100, 125, 300], labels=["normal", "prediabetic", "high"])
    d["BMI"] = pd.cut(d["BMI"], [0, 25, 30, 80], labels=["normal", "overweight", "obese"])
    d["Age"] = pd.cut(d["Age"], [0, 30, 45, 120], labels=["<30", "30-45", "45+"])
    d["BloodPressure"] = pd.cut(d["BloodPressure"], [0, 80, 90, 200], labels=["normal", "elevated", "high"])
    d["Pregnancies"] = pd.cut(d["Pregnancies"], [-1, 1, 5, 30], labels=["0-1", "2-5", "6+"])
    d["Outcome"] = d["Outcome"].map({0: "No", 1: "Yes"})
    items = pd.get_dummies(d[["Glucose", "BMI", "Age", "BloodPressure", "Pregnancies", "Outcome"]].astype(str),
                           prefix_sep="=").astype(bool)
    items = items.loc[:, ~items.columns.str.endswith("=nan")]
    freq = apriori(items, min_support=0.08, use_colnames=True)
    try:
        rules = association_rules(freq, num_itemsets=len(items), metric="confidence", min_threshold=0.6)
    except TypeError:
        rules = association_rules(freq, metric="confidence", min_threshold=0.6)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset({"Outcome=Yes"}))]
    rules = rules.sort_values("lift", ascending=False).head(20)
    for c in ["antecedents", "consequents"]:
        rules[c] = rules[c].apply(lambda s: ", ".join(sorted(s)))
    rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_csv(
        f"{OUT}/association_rules.csv", index=False)
    print(rules[["antecedents", "support", "confidence", "lift"]].head(8))
except ImportError:
    print("mlxtend missing, skipped (pip install mlxtend)")

print("\nDONE. Results are in the outputs folder.")
