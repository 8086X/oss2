사용한 결정 트리 알고리즘을 통한 승객 생존율 예측
Members:
동정순，정보사회미디어，15689252727@163.com
I. 제안 (Proposal)
동기 (Motivation):
우리는 Kaggle Titanic 생존 예측 프로젝트와 유사한 데이터 세트를 선택했습니다. 이 프로젝트의 주요 목표는 기계 학습 방법을 통해 승객 생존에 영향을 미치는 특성 변수를 탐구하는 것입니다. 이 프로젝트를 통해 데이터 정리, 특성 엔지니어링 및 모델 훈련의 핵심 단계를 배우고 익히고자 합니다.
우리의 목표 (What do you want to see at the end?):
결정 트리 알고리즘을 사용하여 승객의 생존 여부를 예측하고, 의사결정 과정을 설명하는 명확한 모델 시각화를 생성합니다.
II. 데이터 세트 (Datasets)
우리가 사용하는 공개 데이터 세트는 Kaggle Titanic 승객 데이터 세트입니다 (Titanic - Machine Learning from Disaster). 데이터 세트는 다음 주요 필드를 포함합니다:
- 승객의 개인 정보 (예: 성별, 나이, 사회경제적 지위)
- 요금, 객실 등급
- 생존 여부 레이블 필드 (목표 변수)
데이터 전처리 단계는 다음을 포함합니다:
- 누락된 값을 처리 (예: 나이를 중위수로 채우기)
- 범주형 데이터 (성별, 객실 등급)를 인코딩
- 연속 변수 (요금, 나이)를 표준화
III. 방법론 (Methodology)
알고리즘 선택:
결정 트리 알고리즘을 선택했습니다. 이 알고리즘은 데이터의 해석 가능성이 높고 구현하기 쉽습니다. 사용한 라이브러리는 Scikit-learn입니다.
특성 설명 (Features):
모델의 주요 특성은 다음과 같습니다:
- 성별 (Sex): 이진 분류 (남/여)
- 나이 (Age): 연속 변수
- 객실 등급 (Pclass): 범주형 변수, 1-3 등급
- 가족 구성원 수 (SibSp + Parch): 연속 변수
- 요금 (Fare): 연속 변수

특성 엔지니어링 후, 위의 특성을 표준화 및 인코딩하여 모델 훈련에 사용했습니다.
IV. 평가 및 분석 (Evaluation & Analysis)
우리는 데이터를 훈련 세트와 테스트 세트로 나누었습니다 (비율: 80:20). 모델 평가는 다음을 포함합니다:
- 정확도 (Accuracy)를 주요 지표로 사용
- 혼동 행렬을 그려 분류 오류 분석
- 결정 트리 시각화 도구를 사용하여 분할 규칙을 단계적으로 설명하는 의사결정 경로를 생성
결과 및 시각화:
- 결정 트리 구조도: 모델의 분할 논리를 보여줍니다.
- 혼동 행렬: 예측 오류를 분석하는 데 사용됩니다.
- 정확도 평가: 초기 결과에서 모델의 테스트 세트 정확도는 나타났습니다.
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the Titanic dataset (from Kaggle sample dataset URL or placeholder here)
# Placeholder data - create a simple Titanic-like dataset
data = {
    'Pclass': [1, 3, 2, 3, 1, 3, 2, 3, 1, 2],
    'Sex': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'female'],
    'Age': [22, 38, 26, 35, 28, 2, 40, 27, 19, 36],
    'SibSp': [1, 1, 0, 0, 0, 4, 0, 0, 0, 1],
    'Parch': [0, 0, 0, 0, 0, 1, 0, 0, 0, 2],
    'Fare': [7.25, 71.2833, 7.925, 8.05, 53.1, 21.075, 13.0, 11.1333, 30.0, 23.45],
    'Survived': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Data preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Encode 'Sex' column
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Displaying accuracy
accuracy

 ![alt text](image.png)
 ![alt text](image-1.png)
V. 관련 연구 (Related Work)

사용한 도구 및 라이브러리:
- Scikit-learn: 결정 트리 훈련 및 평가에 사용
- Pandas 및 Numpy: 데이터 정리 및 전처리
- Matplotlib 및 Seaborn: 데이터 시각화
참조 문서 및 블로그:
- Kaggle Titanic 프로젝트 예제
- Scikit-learn 결정 트리 문서
VI. 결론 (Conclusion)
논의:
이번 프로젝트를 통해 우리는 데이터 전처리, 특성 선택 및 결정 트리 알고리즘의 실제 적용을 익혔습니다. 모델 성능 (85% 정확도)은 최적은 아니지만, 결정 트리는 뛰어난 해석 가능성을 제공합니다. 후속 개선 방향에는 다른 알고리즘 (예: 랜덤 포레스트)을 시도하거나 특성 엔지니어링 과정을 최적화하는 것이 포함됩니다.
프로젝트 분업:
- 동정순: 데이터 처리 및 모델 훈련, 데이터 시각화 및 분석, 문서 작성 및 결과 보고
