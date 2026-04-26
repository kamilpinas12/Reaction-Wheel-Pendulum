# RL Module: Reaction-Wheel-Pendulum

Dokumentacja modułów warstwy RL oraz konfiguracji środowiska developerskiego.

## Moduł src/reac_wheel_sim

Ten moduł odpowiada za pełną warstwę symulacji i generowania danych do uczenia.

Zakres odpowiedzialności:
- definicja dynamiki Reaction Wheel Pendulum i interfejsu zgodnego z Gymnasium,
- randomizacja parametrów fizycznych między epizodami,
- dodawanie szumu obserwacji,
- generowanie sekwencji sygnałów sterujących,
- tworzenie datasetu epizodów do estymacji parametrów.

Pliki i ich rola:
- `RL/src/reac_wheel_sim/reaction_wheel_env.py`
	- klasa środowiska (`ReactionWheelEnv`),
	- definicja przestrzeni akcji i obserwacji,
	- integracja dynamiki (`_rk4_step`),
	- reward shaping i warunki zakończenia,
	- wrappery do randomizacji parametrów i szumu obserwacji.
- `RL/src/reac_wheel_sim/signal_generator.py`
	- klasy generatorów sygnału wejściowego,
	- sygnały deterministyczne i losowe używane podczas generacji epizodów.
- `RL/src/reac_wheel_sim/pend_sim_dataset.py`
	- generowanie epizodów przez wielokrotny rollout środowiska,
	- budowanie tensora cech o stałej długości sekwencji,
	- budowanie targetu parametrów fizycznych,
	- narzędzia pomocnicze do wizualizacji próbek i rozkładu targetów.

Co powinno się znajdować w tym module:
- wyłącznie logika symulacji, sygnałów i danych,
- brak logiki trenowania modeli sieciowych,
- API stabilne dla warstwy `src/nets` i skryptów z `RL/scripts`.

## Moduł src/nets

Ten moduł odpowiada za modele ML i narzędzia ewaluacji predykcji.

Zakres odpowiedzialności:
- definicja architektury estymatora parametrów,
- inferencja na danych sekwencyjnych,
- obliczanie metryk jakości i wizualizacje wyników.

Pliki i ich rola:
- `RL/src/nets/parameter_estimator.py`
	- model `PhysicalParameterEstimator`,
	- architektura sekwencyjna (GRU + head MLP),
	- wejście: sekwencja obserwacji z datasetu,
	- wyjście: wektor estymowanych parametrów.
- `RL/src/nets/visualizations.py`
	- ocena predykcji modelu na zbiorze walidacyjnym/testowym,
	- metryki per-parametr (np. MAE, RMSE, R2),
	- zapisywanie wykresów i raportowanie wyników.

Co powinno się znajdować w tym module:
- logika modelu i oceny jakości predykcji,
- brak logiki fizyki środowiska i generatorów sygnałów,
- brak konfiguracji shell/pytest (to należy do warstwy projektu).

## Konfiguracja Projektu

### src/config.py

Plik `RL/src/config.py` zawiera wspólne ustawienia ścieżek i logowania dla modułu RL.

W tym projekcie definiuje m.in.:
- `PROJECT_ROOT` - root modułu RL,
- `LOGS_DIR` - katalog bazowy logów (`RL/logs`),
- `MODELS_DIR` - katalog modeli (`RL/saved_models`),
- `TB_LOG_DIR` - katalog logów TensorBoard tworzony ze znacznikiem czasu,
- parametry logowania TensorBoard (`TB_FLUSH_SECS`, `TB_COMMENT`).

Zasada pracy:
- wszystkie skrypty treningowe i ewaluacyjne powinny korzystać z wartości z `config.py`,
- nie należy hardkodować ścieżek absolutnych do logów, checkpointów ani modeli,
- nowe ścieżki konfiguracyjne dodajemy najpierw tutaj, a dopiero potem używamy ich w kodzie.

### pyproject.toml

Plik `pyproject.toml` definiuje konfigurację testów dla całego repo.

W tym projekcie zawiera m.in.:
- `testpaths = ["RL/tests"]` - katalog domyślnie przeszukiwany przez pytest,
- `pythonpath = ["RL/src"]` - ścieżka importów modułów RL,
- konfigurację logowania sesji pytest do `RL/logs/pytest_runs.log`,
- domyślne opcje uruchomienia testów (`addopts`).

To jest centralne miejsce na ustawienia testowe i powinno pozostać jedynym źródłem prawdy dla zachowania pytest w repo.

### .env.example

Plik `.env.example` dokumentuje wymagane zmienne środowiskowe dla lokalnego uruchamiania.

W tym projekcie powinien zawierać minimum:
- `PYTHONPATH=RL/src` - poprawne rozwiązywanie importów modułów RL,
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` - izolacja od zewnętrznych pluginów pytest w systemie.

Zasada pracy:
- `.env.example` jest szablonem,
- lokalnie tworzysz `.env` na jego podstawie,
- nowe zmienne wymagane przez kod powinny być dopisywane najpierw do `.env.example`.

## Testy

Testy znajdują się w katalogu `RL/tests` i są uruchamiane przez `pytest`.

Zakres testów:
- testy spójności generacji danych i datasetu,
- testy porównania zachowania symulacji z danymi referencyjnymi MATLAB,
- hooki logujące przebieg sesji testowej do pliku.

Uruchamianie testów z root repo:

```bash
pytest
```

Ważne informacje operacyjne:
- konieczne ustawienie zmiennych z `.env.example`
- domyślny katalog testów jest ustawiony w `pyproject.toml` (`testpaths`),
- logi sesji pytest trafiają do `RL/logs/pytest_runs.log`,
- przy uruchamianiu w środowisku systemowym warto zostawić `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, żeby uniknąć kolizji z globalnymi pluginami.

