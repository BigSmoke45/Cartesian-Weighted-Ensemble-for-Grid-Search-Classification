import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal  # Додали імпорт scipy.signal
def main():
    # Завантаження даних
    file_path = 'A:/AnimalCrosFor/socks.csv'  # Путь к файлу CSV
    data = pd.read_csv(file_path)
    file_name = file_path.split('/')[-1]  # Извлечение имени файла из его пути
    # Вывод имени файла
    print("Working with file:", file_name)
    # Підготовка даних
    X = data.drop(['Name', 'Unique Entry ID'], axis=1)
    y = data['Name']
    # Кодування категоріальних ознак
    X_encoded = pd.get_dummies(X)
    # Кількість повторень навчання моделі
    n_repeats = 12
    # Зберігаємо точність кожного повторення
    accuracies = []
    for _ in range(n_repeats):
        # Розділення даних на навчальний та тестовий набори
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        # Створення ансамблю методом горизонтального голосування з вагами
        ensemble = VotingClassifier(estimators=[('dt', DecisionTreeClassifier()), 
                                                ('lr', make_pipeline(SimpleImputer(strategy='most_frequent'), 
                                                                     LogisticRegression(solver='liblinear')))], 
                                    voting='soft', weights=[0.6, 0.4])  # Задаємо ваги для кожної моделі
        # Навчання моделі
        ensemble.fit(X_train, y_train)
        # Оцінка точності моделі
        accuracy = accuracy_score(y_test, ensemble.predict(X_test))
        accuracies.append(accuracy)
    # Вивід середнього значення та стандартного відхилення точності
    print("Mean Accuracy:", np.mean(accuracies))
    print("Standard Deviation of Accuracy:", np.std(accuracies))
    # Побудова графіка точності
    plt.plot(range(1, n_repeats + 1), accuracies, marker='o')
    plt.xlabel('Repeat Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Repeats')
    plt.grid(True)
    # Вибір пікових значень або значень, що формують стабільне плато
    peaks = find_peaks(accuracies, prominence=0.01)  # Змініть параметр prominence за необхідності
    plt.plot(np.array(peaks) + 1, np.array(accuracies)[peaks], "x", markersize=10, color='r')
    plt.show()
def find_peaks(data, prominence):
    """Функція для пошуку пікових значень"""
    peaks, _ = scipy.signal.find_peaks(data, prominence=prominence)
    return peaks
if __name__ == "__main__":
    main()