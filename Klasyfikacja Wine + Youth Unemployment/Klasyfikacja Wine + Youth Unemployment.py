"""
   Projekt: Klasyfikacja Wine + Youth Unemployment (World Bank API)

AUTORZY:
Oleksii Sumrii (s22775),
Oskar Szyszko (s28838),

INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA:
Zainstaluj Python 3.8 lub nowszy z https://python.org/,
Zainstaluj Visual Studio Code (VS Code) z https://code.visualstudio.com/,

Zainstaluj wymagane biblioteki w terminalu (po zainstalowaniu Pythona), wpisz:
    pip install pandas seaborn matplotlib scikit-learn requests

Zainstaluj rozszerzenie „Python” w VS Code:
    Otwórz VS Code,
    Kliknij ikonę rozszerzeń po lewej stronie albo Ctrl+Shift+X,
    Wyszukaj "Python" i zainstaluj rozszerzenie od Microsoft.

PRZYGOTOWANIE I URUCHOMIENIE PROJEKTU:

1)Otwórz w VS Code folder, w którym znajduje się plik z tym kodem (np. world_bank_unemployment.py).
2)Upewnij się, że masz dostęp do internetu:
    – część kodu korzysta z API World Bank do pobierania danych o bezrobociu młodych.,
3)Uruchom skrypt:
    – Otwórz plik .py z kodem,
    – Kliknij przycisk "Run Python File" w prawym górnym rogu
    LUB użyj skrótu klawiszowego Ctrl+F5.,
,

DZIAŁANIE PROGRAMU (SKRÓTOWO):
    CZĘŚĆ 1:
        – Ładuje wbudowany zbiór danych Wine z biblioteki scikit-learn,
        – Trenuje klasyfikator Drzewa Decyzyjnego i SVM,
        – Wypisuje dokładność modeli dla zbioru Wine.

    CZĘŚĆ 2:
        – Pobiera dane o bezrobociu młodych (15–24 lata) z World Bank API,
        – Przetwarza dane do postaci tabeli (pivot) z latami jako kolumnami,
        – Tworzy zmienną klasową (wysokie / niskie bezrobocie w 2023 na podstawie mediany),
        – Dzieli dane na zbiór treningowy i testowy, skaluje cechy,
        – Trenuje Drzewo Decyzyjne i kilka modeli SVM (różne kernele),
        – Wypisuje dokładność modeli oraz wykonuje test na przykładowych danych.

"""

import pandas as pd  # pd -> alias (skrót) dla biblioteki pandas. Służy do operacji na tabelach (DataFrame).
import seaborn as sns  # sns -> biblioteka do ładnych wykresów statystycznych.
import matplotlib.pyplot as plt  # plt -> podstawowa biblioteka do rysowania wykresów.
import requests  # requests -> biblioteka do komunikacji z internetem (pobieranie danych z API).
from sklearn import (
    datasets,
)  # datasets -> magazyn gotowych zbiorów danych w Scikit-learn.
from sklearn.model_selection import (
    train_test_split,
)  # funkcja do dzielenia danych na treningowe i testowe.
from sklearn.tree import DecisionTreeClassifier  # Klasa algorytmu Drzewa Decyzyjnego.
from sklearn.svm import SVC  # Klasa algorytmu SVM (Support Vector Classifier).
from sklearn.metrics import (
    accuracy_score,
)  # Funkcja do liczenia dokładności (ile % trafionych).
from sklearn.preprocessing import (
    StandardScaler,
)  # Narzędzie do skalowania danych (standaryzacja).

# CZĘŚĆ 1: KLASYFIKACJA WINE DATASET (BAZOWA)
print("--- CZĘŚĆ 1: KLASYFIKACJA WINE DATASET ---")
# print -> wypisanie tekstu.
# string w "" -> tekst wypisywany na konsolę.

wine = (
    datasets.load_wine()
)  # datasets to specjalny moduł wewnątrz biblioteki Scikit-learn, który działa jak magazyn z przykładowymi danymi.
# Funkcja .load_wine() to specjalne polecenie z biblioteki Scikit-learn, które służy do natychmiastowego pobrania gotowego, "laboratoryjnego" zbioru danych o winach.
# datasets.load_wine() -> ładuje zbiór win.
"""
Typ Bunch to specjalny obiekt (kontener) używany w bibliotece scikit-learn, który służy do przechowywania zbiorów danych.
Od zwykłego słownika on się różni tym że nie używa nawiasów i cudzysłowów np. słownik['klucz'].
Bunch pozwala na dostęp do tych samych danych za pomocą kropki, co jest znacznie wygodniejsze i czytelniejsze w kodzie.
"""

X_wine = wine.data  # wine.data -> macierz cech.
# Macierz z liczbami (cechy: alkohol, kwasowość itp.) – to przypisujemy do X.
# wine.data to surowe dane pomiarowe na których nasz model będzie się uczył.
# te dane np. Alkochol = 14, Kwas = 2.
y_wine = wine.target  # y_wine -> etykiety klas.
# wine.target -> wektor np. [0,1,2].
# .target odpowiada za klass wina można jeszcze nazwać etykietą (do jakiego klasu będzie się odnosiło)

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42
)
# = -> przypisanie wartości (rozpakowanie wyników funkcji do 4 zmiennych).
# train_test_split -> dzieli dane na część treningową i testową.
# X_wine -> cechy wejściowe (dane).
# y_wine -> etykiety (odpowiedzi).
# test_size=0.3 -> 30% danych idzie do testu (egzamin).
# do treningu automatycznie bierze 70 procent (nauka), bo do testu daliśmy 30.
# random_state=42 -> powtarzalność losowania.
"""
Gdy wywołujesz funkcję z parametrami test_size=0.3 i random_state=42:
Tasowanie (Shuffling): Funkcja najpierw miesza wiersze w tabeli jak talię kart 
(dzięki temu wino klasy 0, 1 i 2 jest rozłożone równomiernie, a nie pogrupowane). 
Odpowiada za to domyślny parametr shuffle=True.
Cięcie: Następnie "odcina" 30% kart z góry talii i odkłada na bok jako Zbiór Testowy.
Reszta: Pozostałe 70% kart zostaje jako Zbiór Treningowy.
Podział pionowy: Na koniec rozdziela karty na cechy (X) i wyniki (y), 
zachowując ich dopasowanie (żeby wynik nadal pasował do cech).
"""

#  DRZEWO DECYZYJNE 
dt_wine = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
# DecisionTreeClassifier -> konstruktor tworzący obiekt modelu (puste drzewo).
# criterion='entropy' -> Mówi drzewu, żeby używało wzoru na Entropię Shannona.
# max_depth=3 -> Ogranicza drzewo do 3 poziomów głębokości.
# random_state=42 -> Zapewnia stałość wyników przy budowie drzewa.

"""
Entropia (criterion='entropy'):
[cite_start]Jest to miara nieuporządkowania, losowości lub niepewności w systemie[cite: 33, 43].
Wzór Shannona: H = -sum(p * log(p)).
Drzewo wybiera takie pytania (podziały), które dają największy Zysk Informacyjny (Information Gain),
[cite_start]czyli najbardziej zmniejszają entropię (zwiększają porządek w grupach) [cite: 150-151].

Max_depth (Przycinanie/Pruning):
[cite_start]Ograniczenie głębokości to forma "Pre-pruning" (wstępnego przycinania)[cite: 167].
Zapobiega to zjawisku przeuczenia (overfitting), gdzie drzewo staje się zbyt skomplikowane 
[cite_start]i dopasowuje się do szumu w danych zamiast do ogólnych reguł[cite: 162].
"""

dt_wine.fit(X_train_w, y_train_w)
# .fit -> Metoda "Trenuj". To faza indukcji drzewa.
# [cite_start]Algorytm analizuje X_train_w i y_train_w, budując strukturę pytań od korzenia do liści[cite: 146].

acc_dt = accuracy_score(y_test_w, dt_wine.predict(X_test_w))
# .predict(X_test_w) -> Model dostaje "arkusz egzaminacyjny" (same cechy) i zgaduje odpowiedzi.
# accuracy_score -> Porównuje zgadywanki modelu z prawdziwymi odpowiedziami (y_test_w).
# Zwraca ułamek (np. 0.95), który zapisujemy do zmiennej acc_dt.

print(f"Drzewo Decyzyjne (Wine Accuracy): {acc_dt:.2f}")
# f"..." -> f-string, pozwala wstawić zmienną w tekst.
# :.2f -> formatowanie liczby, pokaż tylko 2 miejsca po przecinku.

# SVM (SUPPORT VECTOR MACHINE)
svm_wine = SVC(kernel="rbf", random_state=42)
# SVC -> Klasa klasyfikatora SVM.
# kernel='rbf' -> Wybór funkcji jądra (Radial Basis Function - Gaussa).
# Używamy go, gdy danych nie da się oddzielić prostą kreską.

svm_wine.fit(X_train_w, y_train_w)
# .fit -> SVM uczy się, szukając "najszerszej ulicy" (marginesu) między klasami.

acc_svm = accuracy_score(y_test_w, svm_wine.predict(X_test_w))
# Obliczenie dokładności dla SVM, analogicznie jak wyżej.

print(f"SVM (Wine Accuracy): {acc_svm:.2f}")

# CZĘŚĆ 2: GLOBAL YOUTH UNEMPLOYMENT (REALNE DANE Z API)
print("\n\n--- CZĘŚĆ 2: YOUTH UNEMPLOYMENT (WORLD BANK API) ---")

# 1. POBIERANIE DANYCH
url = "http://api.worldbank.org/v2/country/all/indicator/SL.UEM.1524.ZS?format=json&per_page=20000"
# url -> zmienna tekstowa z adresem API.
# ?format=json -> prośba o format JSON.
# per_page=20000 -> prośba o maksymalną liczbę wyników na raz.

print(f"Pobieranie danych z: {url} ...")
try:
    response = requests.get(url)
    # requests.get -> wysyła zapytanie do serwera World Bank.
    # response -> obiekt, który przechowuje odpowiedź serwera (kod statusu i dane).

    if response.status_code != 200:
        # sprawdzenie czy status jest OK (200 oznacza sukces).
        print("Błąd połączenia z API!")
        exit()

    data = response.json()
    # .json() -> Tłumaczy odpowiedź serwera (tekst) na struktury Pythona (listy i słowniki).

    # API zwraca listę dwuelementową: [metadane, dane]. Interesuje nas drugi element (indeks 1).
    df = pd.json_normalize(data[1])
    # pd.json_normalize -> Spłaszcza zagnieżdżony JSON (np. słownik w słowniku) do płaskiej tabeli Excela.

except Exception as e:
    # Blok awaryjny - jeśli cokolwiek pójdzie nie tak, wypisz błąd i zakończ.
    print(f"Wystąpił błąd: {e}")
    exit()

# Wybór i zmiana nazw kolumn
df = df[["country.value", "country.id", "date", "value"]]
# [[...]] -> Wybiera tylko te 4 kolumny, resztę wyrzuca.

df.rename(
    columns={
        "country.value": "Country",  # Zmiana nazwy na 'Country'
        "country.id": "CountryCode",  # Zmiana na 'CountryCode'
        "date": "Year",  # Zmiana na 'Year'
        "value": "YouthUnemployment",  # Zmiana na 'YouthUnemployment'
    },
    inplace=True,
)
# inplace=True -> Zapisuje zmiany w tej samej zmiennej df, bez tworzenia kopii.

print(f"Pobrano {len(df)} wierszy.")

# 2. TRANSFORMACJA DANYCH (PIVOT)
print("Przetwarzanie danych...")

df_pivot = df.pivot(index="Country", columns="Year", values="YouthUnemployment")
# .pivot -> Obraca tabelę.
# index='Country' -> Kraje stają się wierszami.
# columns='Year' -> Lata (2020, 2021...) stają się kolumnami.
# values -> W środku tabeli lądują procenty bezrobocia.
"""
Dlaczego Pivot?
Algorytmy ML wymagają, aby jeden wiersz reprezentował jeden obiekt (kraj) z kompletem cech.
Przed pivotem jeden kraj zajmował wiele wierszy (po jednym na rok).
Po pivocie mamy jeden wiersz per kraj, a lata są cechami (Feature 1, Feature 2...).
"""

features = ["2020", "2021", "2022"]  # Cechy: Historia bezrobocia z 3 lat.
target_year = "2023"  # Cel: Rok, który chcemy przewidzieć.

df_model = df_pivot[features + [target_year]].dropna()
# [features + [target_year]] -> Wybiera tylko kolumny z lat 2020-2023.
# .dropna() -> Usuwa wiersze, które mają puste pola (NaN).
# [cite_start]Jest to kluczowe, bo Drzewa i SVM w scikit-learn nie obsługują brakujących danych[cite: 132].

print(f"Liczba krajów w modelu: {len(df_model)}")

# 3. INŻYNIERIA CECH (REGRESJA -> KLASYFIKACJA)
median_val = df_model[target_year].median()
# .median() -> Oblicza wartość środkową dla roku 2023.
print(f"Mediana bezrobocia w {target_year}: {median_val:.2f}% (Punkt podziału)")

df_model["Target_Class"] = (df_model[target_year] > median_val).astype(int)
# Tworzymy nową kolumnę z klasą (0 lub 1).
# ( ... > median_val) -> Zwraca True (jeśli wysokie) lub False (jeśli niskie).
# .astype(int) -> Zamienia True na 1, a False na 0.

X = df_model[features]  # X -> Nasze dane wejściowe (lata 2020-22).
y = df_model["Target_Class"]  # y -> Nasz cel (klasa w 2023).

# Wizualizacja danych
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df_model["2022"],
    y=df_model[target_year],
    hue=df_model["Target_Class"],
    palette="coolwarm",
)
# sns.scatterplot -> Rysuje wykres punktowy.
# hue -> Koloruje punkty w zależności od klasy (0=niebieski, 1=czerwony).
plt.title(f"Bezrobocie: 2022 vs {target_year} (Podział na klasy)")
plt.plot([0, max(df_model["2022"])], [median_val, median_val], "k--", label="Mediana")
# plt.plot -> Rysuje czarną przerywaną linię oznaczającą próg podziału.
plt.legend()
plt.show()  # Wyświetlenie okna z wykresem.


# 4. PODZIAŁ I SKALOWANIE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Ponowne użycie funkcji do podziału danych na treningowe i testowe.

scaler = StandardScaler()
# Inicjalizacja skalera. SVM działa na odległościach geometrycznych.
# Jeśli dane mają różne skale, SVM działa źle. Standaryzacja sprowadza dane do średniej=0 i odchylenia=1.

X_train = scaler.fit_transform(X_train)
# .fit_transform -> Oblicz średnią z danych treningowych I przeskaluj je.

X_test = scaler.transform(X_test)
# .transform -> Przeskaluj dane testowe, używając średniej wyliczonej na treningu.
# WAŻNE: Nie robimy tu fit, żeby nie "podglądać" danych testowych.

# 5. TRENING DRZEWA DECYZYJNEGO (DANE REALNE)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
dt.fit(X_train, y_train)
# Trenujemy drzewo na danych o bezrobociu.

acc = accuracy_score(y_test, dt.predict(X_test))
print(f"\n>>> Drzewo Decyzyjne (Unemployment Accuracy): {acc:.2f}")

# 6. TRENING SVM Z RÓŻNYMI KERNELAMI
kernels = ["linear", "poly", "rbf"]
# Lista trzech rodzajów jąder (soczewek) dla SVM.

print("\n>>> Porównanie Kerneli SVM:")

for k in kernels:
    # Pętla for -> wykonaj ten kod dla każdego elementu z listy kernels.
    # k -> w każdej pętli przyjmuje inną nazwę kernela.

    model = SVC(kernel=k, C=1.0, random_state=42)
    # Tworzymy model SVM z aktualnym kernelem k.

    model.fit(X_train, y_train)
    # Trenujemy model.

    acc = accuracy_score(y_test, model.predict(X_test))
    # Sprawdzamy dokładność.

    print(f"Kernel: {k.ljust(10)} | Accuracy: {acc:.4f}")
    # Wypisujemy wynik. .ljust(10) wyrównuje tekst, żeby tabelka była ładna.

# 7. PRZYKŁADOWE WYWOŁANIE (TEST SYMULACJA)
print("\n[TEST] Przewidywanie dla przykładowego kraju")
# Symulacja kraju, który miał bezrobocie: 10%, 12%, 15%.
sample_data = [[10.0, 12.0, 15.0]]

# Musimy przeskalować te dane tak samo jak dane treningowe!
sample_scaled = scaler.transform(sample_data)

pred_dt = dt.predict(sample_scaled)[0]
# [0] -> wyciągamy wynik (0 lub 1) z listy.

status = "WYSOKIE" if pred_dt == 1 else "NISKIE"
# Prosta instrukcja warunkowa w jednej linii do opisu tekstowego.

print(f"Historia bezrobocia: {sample_data[0]}")
print(f"Przewidywany poziom w {target_year}: {status} (powyżej światowej mediany)")
